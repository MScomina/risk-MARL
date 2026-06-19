# https://arxiv.org/pdf/1312.6120 (Exact solutions to the nonlinear dynamics of learning in deep linear neural networks)

from networkx import DiGraph
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from environment.risk_utils import TradeChoices, TroopActions, RiskPhase
from torch_geometric.nn import Sequential as PyGSequential
from torch_geometric.nn.aggr import MultiAggregation

from .normalization import PureLayerNorm, RunningMeanStd
from .blocks import FiLMBlock, ResidualBlock, GraphResidualBlock, EdgeFeatureExtractor, NodeFeatureExtractor

    
class GraphNetwork(nn.Module):

    def __init__(self, 
                obs_space : spaces.Dict, 
                action_space : spaces.Discrete | int, 
                map_graph : DiGraph, 
                gnn_hidden_size : int = 64,
                res_hidden_size : int = 128,
                activation_function : nn.Module = nn.ReLU,
                embed_space_continent : int = 6,
                embed_space_phase : int = 16,
                graph_depth : int = 2,
                residual_depth : int = 3,
                starting_residual_scale : float = 1.0,
                n_heads : int = 8
                ):
        super().__init__()
        if "observation" in obs_space.keys():
            obs_space = obs_space["observation"]

        self.map_graph : DiGraph = map_graph
        
        self.gnn_hidden_s = gnn_hidden_size
        self.res_hidden_s = res_hidden_size
        self.activation_function = activation_function
        self.embed_space_phase = embed_space_phase
        self.continent_dim = embed_space_continent
        self.graph_depth = graph_depth
        self.residual_depth = residual_depth
        self.starting_residual_scale = starting_residual_scale
        self.n_heads = n_heads
        self.is_likely_critic = (self.output_shape == 1)
        self.n_players = int(obs_space["territory_owner"].high[0])

        self.node_feature_extractor = NodeFeatureExtractor(
            map_graph=self.map_graph,
            n_players=self.n_players,
            embed_space_continent=self.continent_dim,
            embed_space_phase=self.embed_space_phase
        )

        self.edge_feature_extractor = EdgeFeatureExtractor(
            map_graph=self.map_graph
        )

        if isinstance(action_space, spaces.Discrete):
            self.output_shape = int(action_space.n)
        else:
            self.output_shape = action_space

        f_in = self.node_feature_extractor.get_num_parameters()

        gnn_layers = [
            (
                GraphResidualBlock(
                    f_in,
                    self.gnn_hidden_s,
                    self.n_heads,
                    activation_fn=self.activation_function,
                    dropout=0.1,
                    is_residual=True,
                ),
                'x, edge_index, edge_attr -> x'
            )
        ]

        gnn_layers.extend(
            (
                GraphResidualBlock(
                    self.gnn_hidden_s,
                    self.gnn_hidden_s,
                    self.n_heads,
                    activation_fn=self.activation_function,
                    dropout=0.1,
                    is_residual=True,
                ),
                'x, edge_index, edge_attr -> x'
            )
            for _ in range(self.graph_depth - 1)
        )

        self.gnn = PyGSequential(
            'x, edge_index, edge_attr',
            gnn_layers
        )

        self.pool_layers_aggrs = ["mean", "min", "max", "std", "sum"]
        self.pool_layers_aggrs_kwargs = [
            {},
            {},
            {},
            {},
            {}
        ]

        self.pool_layers = PyGSequential(
            "x, index",
            [
                (MultiAggregation(
                    aggrs=self.pool_layers_aggrs,
                    aggrs_kwargs=self.pool_layers_aggrs_kwargs,
                    mode="cat"
                ), "x, index -> x"),
                (PureLayerNorm(len(self.pool_layers_aggrs)*self.gnn_hidden_s), "x -> x")
            ]
        )

        self.node_score_head = nn.Sequential(
            nn.Linear(
                self.gnn_hidden_s + self.res_hidden_s,
                self.gnn_hidden_s
            ),
            activation_function(),
            nn.Linear(self.gnn_hidden_s, 1)
        )
        self.edge_attr_dim = 5
        self.edge_score_head = nn.Sequential(
            nn.Linear(
                (self.gnn_hidden_s * 2)
                + self.edge_attr_dim
                + self.res_hidden_s,
                self.gnn_hidden_s
            ),
            activation_function(),
            nn.Linear(self.gnn_hidden_s, 1)
        )
        parameter_out_shape = max(len(TradeChoices), len(TroopActions)) if not self.is_likely_critic else 1
        self.parameter_head = nn.Linear(self.res_hidden_s, parameter_out_shape)

        projection_size = (len(self.pool_layers_aggrs)*self.gnn_hidden_s) + 1

        if not self.is_likely_critic:
            self.film_block = FiLMBlock(
                input_shape=self.embed_space_phase,
                output_shape=self.res_hidden_s
            )

        self.residual_block = nn.Sequential(
            *[ResidualBlock(
                channels=self.res_hidden_s,
                activation_fn=self.activation_function,
                starting_residual_scale=self.starting_residual_scale
            ) for _ in range(self.residual_depth)]
        )

        if "cards_in_hand" in obs_space.keys():
            projection_size += obs_space["cards_in_hand"].shape[0] + 3
            self.cards_norm = RunningMeanStd(
                obs_space["cards_in_hand"].shape[0]
            )

            self.other_cards_norm = RunningMeanStd(3)
            
        self.global_projection = nn.Linear(projection_size, self.res_hidden_s)

        self.norm_logits = PureLayerNorm(self.res_hidden_s)

        self.reinforcement_norm = RunningMeanStd(1)

        self._init_weights()

    def forward(self, obs, state=None, info={}):
        
        device = next(self.parameters()).device

        if "obs" in obs:
            obs_dict = obs["obs"]
        else:
            obs_dict = obs

        obs_clean = {
            "territory_owner" : torch.as_tensor(obs_dict["territory_owner"], dtype=torch.long, device=device),
            "selected_node" : torch.as_tensor(obs_dict["selected_node"], dtype=torch.long, device=device),
            "selected_edge" : torch.as_tensor(obs_dict["selected_edge"], dtype=torch.long, device=device),
            "action_phase" : torch.as_tensor(obs_dict["action_phase"], dtype=torch.long, device=device),
            "number_of_armies" : torch.as_tensor(obs_dict["number_of_armies"], dtype=torch.float32, device=device)
        }

        batch_size = obs_clean["territory_owner"].shape[0]
        num_nodes = obs_clean["territory_owner"].shape[1]

        node_features, normalized_armies = self.node_feature_extractor(list(obs_clean.values()))

        batched_edge_index, edge_attr = self.edge_feature_extractor(list(obs_clean.values()), normalized_armies)

        x_flat = node_features.view(-1, node_features.shape[-1])

        h_nodes = self.gnn(x_flat, batched_edge_index, edge_attr)

        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
        h_pooled = self.pool_layers(h_nodes, index=batch_idx)

        reinforce_tensor = torch.as_tensor(obs_dict["troops_to_place"], dtype=torch.float32, device=device)
        normalized_reinforcements = self.reinforcement_norm(reinforce_tensor.view(-1, 1))

        if "cards_in_hand" in obs_dict:

            raw_cards = torch.as_tensor(obs_dict["cards_in_hand"], dtype=torch.float32, device=device)
            raw_hands = torch.as_tensor(obs_dict["amount_cards_others"], dtype=torch.float32, device=device)
            cards_normalized = self.cards_norm(torch.log1p(raw_cards))
            hands_normalized = self.other_cards_norm(
                torch.stack([
                    torch.log1p(raw_hands.max(-1).values),
                    torch.log1p(raw_hands.median(-1).values),
                    torch.log1p(raw_hands.sum(-1))], 
                dim=-1)
            )
            
        else:
            cards_normalized = torch.zeros((batch_size, 0), dtype=torch.float32, device=device)
            hands_normalized = torch.zeros((batch_size, 0), dtype=torch.float32, device=device)

        h_global = torch.cat([
            h_pooled,
            normalized_reinforcements,
            cards_normalized,
            hands_normalized
        ], dim=-1)
        h_global = self.activation_function()(self.global_projection(h_global))

        if not self.is_likely_critic:
            h_global = self.film_block(h_global, cond=self.node_feature_extractor.phase_embedder(obs_clean["action_phase"]))

        h_res = self.residual_block(h_global)
        h_res = self.norm_logits(h_res)

        full_logits = torch.full((batch_size, self.output_shape), float('-inf'), device=device)
        unique_phases = obs_clean["action_phase"].unique()

        if not self.is_likely_critic:
            for phase in unique_phases:
                phase_mask = (obs_clean["action_phase"] == phase)
                num_in_phase = phase_mask.sum().item()
                if phase in [RiskPhase.SELECT_NODE, RiskPhase.STARTING_PLACEMENT]:
                    h_nodes_b = h_nodes.view(batch_size, num_nodes, -1)

                    global_context = (
                        h_res
                        .unsqueeze(1)
                        .expand(-1, num_nodes, -1)
                    )

                    node_features = torch.cat(
                        [h_nodes_b, global_context],
                        dim=-1
                    )

                    phase_scores = self.node_score_head(node_features)
                    phase_scores = phase_scores.squeeze(-1)
                    phase_scores = phase_scores.view(batch_size, -1)

                    full_logits[phase_mask, phase_scores.shape[1]] = -1e6   # Arbitrarily large but not infinite logit for the No-op operation
                    full_logits[phase_mask, :phase_scores.shape[1]] = phase_scores[phase_mask]

                elif phase in [RiskPhase.SELECT_EDGE]:

                    edge_src_b, edge_dst_b = batched_edge_index
                    h_src = h_nodes[edge_src_b]
                    h_dst = h_nodes[edge_dst_b]    
                
                    num_edges = edge_index.shape[1]

                    edge_global = (
                        h_res
                        .unsqueeze(1)
                        .expand(-1, num_edges, -1)
                        .reshape(-1, self.res_hidden_s)
                    )

                    edge_features = torch.cat(
                        [
                            h_src,
                            h_dst,
                            edge_attr,
                            edge_global
                        ],
                        dim=-1
                    )
                    phase_scores = self.edge_score_head(edge_features.view(batch_size, -1, self.gnn_hidden_s*2+self.edge_attr_dim+self.res_hidden_s))
                    phase_scores = phase_scores.view(batch_size, -1)

                    full_logits[phase_mask, phase_scores.shape[1]] = -1e6   # Arbitrarily large but not infinite logit for the No-op operation
                    full_logits[phase_mask, :phase_scores.shape[1]] = phase_scores[phase_mask]
                else:
                    phase_scores = self.parameter_head(h_res)
                    full_logits[phase_mask, :phase_scores.shape[1]] = phase_scores[phase_mask]
        else:
            phase_scores = self.parameter_head(h_res)
            full_logits = phase_scores

        logits = full_logits

        if not self.is_likely_critic and "mask" in obs:
            valid_mask = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device)
            logits = logits.masked_fill(~valid_mask, float('-inf'))
            logits = logits - logits.max(dim=-1, keepdim=True).values

        if self.is_likely_critic:
            return logits.cpu()

        return logits.cpu(), state


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

        for module_list in [self.node_score_head.modules(), self.edge_score_head.modules()]:
            for m in module_list:
                if isinstance(m, nn.Linear):
                    torch.nn.init.orthogonal_(m.weight, gain=0.05)
                    torch.nn.init.zeros_(m.bias)
