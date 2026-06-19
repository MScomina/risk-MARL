# https://arxiv.org/pdf/1312.6120 (Exact solutions to the nonlinear dynamics of learning in deep linear neural networks)
# https://arxiv.org/pdf/1709.07871 (FiLM: Visual Reasoning with a General Conditioning Layer)

from networkx import DiGraph
import torch
import torch_geometric as torchg
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from environment.risk_utils import TradeChoices, TroopActions, RiskPhase
from tianshou.data.batch import Batch
from tianshou.algorithm.algorithm_base import Policy
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE, GATv2Conv, TransformerConv, Sequential as PyGSequential

class RunningMeanStd(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-4):
        super().__init__()

        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("var", torch.ones(feature_dim))
        self.register_buffer("count", torch.tensor(eps))


    @torch.no_grad()
    def update(self, x: torch.Tensor):

        x = x.float()

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean

        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count

        M2 = (
            m_a
            + m_b
            + delta.pow(2)
            * self.count
            * batch_count
            / total_count
        )

        self.mean.copy_(new_mean)
        self.var.copy_(M2 / total_count)
        self.count.copy_(total_count)

    def forward(self, x: torch.Tensor):

        if self.training:
            self.update(x.reshape(-1, x.shape[-1]))
            
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class PureLayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale (gamma) and shift (beta) parameters
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        orig_dtype = x.dtype
        x = x.float()

        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)

        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        out = x_norm * self.weight.float() + self.bias.float()

        return out.to(orig_dtype)

class ResidualBlock(nn.Module):

    def __init__(self, channels, activation_fn : nn.Module = nn.ReLU, starting_residual_scale : float = 1.0, is_residual : bool = True):
        super().__init__()
        self.is_residual = is_residual
        self.activation = activation_fn()
        self.norm = PureLayerNorm(channels)
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            self.activation,
            nn.Linear(channels, channels)
        )
        if self.is_residual:
            self.residual_scale = nn.Parameter(torch.tensor(starting_residual_scale))

    def forward(self, x):
        x_norm = self.norm(x)
        if self.is_residual:
            return x + self.residual_scale * self.block(x_norm)
        else:
            return self.block(x_norm)


class GraphResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_size,
        n_heads,
        activation_fn=nn.ReLU,
        edge_dim=5,
        dropout=0.1,
        is_residual=True,
        starting_residual_scale=1.0
    ):
        super().__init__()

        self.is_residual = is_residual and (in_channels == hidden_size)

        self.conv = TransformerConv(
            in_channels,
            hidden_size,
            heads=n_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        self.proj = nn.Linear(
            hidden_size * n_heads,
            hidden_size
        )

        self.norm = PureLayerNorm(hidden_size)
        self.activation = activation_fn()
        if self.is_residual:
            self.residual_scale = nn.Parameter(torch.tensor(starting_residual_scale))

    def forward(self, x, edge_index, edge_attr):

        residual = x

        x = self.conv(x, edge_index, edge_attr)
        x = self.proj(x)
        x = self.norm(x)
        x = self.activation(x)

        if self.is_residual:
            x = (self.residual_scale * x) + residual

        return x

    
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
        self.continent_dim = embed_space_continent
        self.graph_depth = graph_depth
        self.residual_depth = residual_depth
        self.starting_residual_scale = starting_residual_scale
        self.n_heads = n_heads

        unique_continents = sorted(set([
            self.map_graph.nodes[node].get('continent', 'Unknown')
            for node in self.map_graph.nodes
        ]))
        self.continent_to_idx = {name: i for i, name in enumerate(unique_continents)}
        self.continent_embedder = nn.Embedding(20, self.continent_dim)

        self.num_owner_categories = 3     # Not owned, owned by myself, owned by someone else.
        self.embed_space_phase = embed_space_phase

        if isinstance(action_space, spaces.Discrete):
            self.output_shape = int(action_space.n)
        else:
            self.output_shape = action_space
        self.is_likely_critic = (self.output_shape == 1)
        self.n_players = int(obs_space["territory_owner"].high[0])

        f_in = (
            self.num_owner_categories +
            1 + # Normalized armies
            1 + # Owner army total
            1 + # Owner territory total
            self.embed_space_phase + 
            self.continent_dim +
            1 + # Node selected one-hot
            1 # Edge selected indicator
        )

        self.phase_embedder = nn.Embedding(
            num_embeddings=len(RiskPhase), 
            embedding_dim=self.embed_space_phase
        )

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
                (torchg.nn.aggr.MultiAggregation(
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

        projection_size = (len(self.pool_layers_aggrs)*self.gnn_hidden_s) + 1 # pooling layers and reinforcements

        # FiLM implementation
        self.phase_gamma = nn.Linear(self.embed_space_phase, self.res_hidden_s)
        self.phase_beta  = nn.Linear(self.embed_space_phase, self.res_hidden_s)

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

        self.army_norm = RunningMeanStd(1)
        self.owner_army_norm = RunningMeanStd(1)
        self.owner_territory_norm = RunningMeanStd(1)
        self.reinforcement_norm = RunningMeanStd(1)

        self._init_weights()

    def forward(self, obs, state=None, info={}):
        
        device = next(self.parameters()).device

        if "obs" in obs:
            obs_dict = obs["obs"]
        else:
            obs_dict = obs

        # Basic initialization
        territory_owner = torch.as_tensor(obs_dict["territory_owner"], dtype=torch.long, device=device)
        selected_node = torch.as_tensor(obs_dict["selected_node"], dtype=torch.long, device=device)
        selected_edge = torch.as_tensor(obs_dict["selected_edge"], dtype=torch.long, device=device)
        action_phase = torch.as_tensor(obs_dict["action_phase"], dtype=torch.long, device=device)
        armies_tensor = torch.as_tensor(obs_dict["number_of_armies"], dtype=torch.float32, device=device)

        int_edges = [(self.map_graph.graph["node_to_idx"][u], self.map_graph.graph["node_to_idx"][v])
                     for u, v in self.map_graph.edges]
        edge_index = torch.tensor(list(int_edges), dtype=torch.long, device=device).t().contiguous()

        batch_size = territory_owner.shape[0]
        num_nodes = territory_owner.shape[1]

        node_continents = []
        for i in range(num_nodes):
            node_name = self.map_graph.graph["idx_to_node"][i]
            cont_name = self.map_graph.nodes[node_name]["continent"]
            node_continents.append(self.continent_to_idx[cont_name])

        continent_indices = torch.tensor(node_continents, dtype=torch.long, device=device)
        continent_embeddings = self.continent_embedder(continent_indices).unsqueeze(0).expand(batch_size, -1, -1)

        phase_emb = self.phase_embedder(action_phase)
        phase_node_context = phase_emb.unsqueeze(1).repeat(1, num_nodes, 1)

        owner_category = torch.where(
            territory_owner > 0,
            2,
            torch.where(territory_owner == 0, 1, 0)
        )
        owner_embedding = torch.nn.functional.one_hot(owner_category, num_classes=self.num_owner_categories)        

        normalized_armies = self.army_norm(
            torch.log1p(armies_tensor.unsqueeze(-1))
        )

        owner_shifted = territory_owner + 1

        army_by_owner = torch.zeros(batch_size, self.n_players + 1, device=device, dtype=torch.float)
        territories_by_owner = torch.zeros(batch_size, self.n_players + 1, device=device, dtype=torch.float)

        army_by_owner.scatter_add_(1, owner_shifted, armies_tensor)
        territories_by_owner.scatter_add_(
            1,
            owner_shifted,
            torch.ones_like(owner_shifted, dtype=torch.float)
        )

        owner_army_total = self.owner_army_norm(torch.log1p(army_by_owner.gather(1, owner_shifted)).unsqueeze(-1))
        owner_territory_total = territories_by_owner.gather(1, owner_shifted)

        owner_territory_total = self.owner_territory_norm(
            owner_territory_total.unsqueeze(-1)
        )

        node_select_onehot = torch.zeros(batch_size, num_nodes, device=device)
        valid_node_sel = selected_node >= 0
        node_select_onehot[valid_node_sel, selected_node[valid_node_sel]] = 1.0
        node_select_onehot = node_select_onehot.unsqueeze(2)

        edge_node_indicator = torch.zeros(batch_size, num_nodes, device=device)
        valid_edge_sel = selected_edge >= 0

        if valid_edge_sel.any():
            active_edges = selected_edge[valid_edge_sel]
            src_nodes = edge_index[0, active_edges]
            tgt_nodes = edge_index[1, active_edges]

            batch_indices = torch.arange(batch_size, device=device)[valid_edge_sel]
            edge_node_indicator[batch_indices, src_nodes] = -1.0
            edge_node_indicator[batch_indices, tgt_nodes] = 1.0

        edge_node_indicator = edge_node_indicator.unsqueeze(2)

        features_array = [
            owner_embedding,
            normalized_armies,
            owner_army_total,
            owner_territory_total,
            phase_node_context,
            continent_embeddings,
            node_select_onehot,
            edge_node_indicator
        ]

        # Computing node and edge features for GNN
        node_features = torch.cat(features_array, dim=2)

        edge_src, edge_dst = edge_index
        batch_offsets = torch.arange(batch_size, device=device) * num_nodes

        flat_armies = normalized_armies.squeeze(-1).view(-1)
        flat_owners = territory_owner.reshape(-1)
        flat_raw_armies = armies_tensor.reshape(-1)

        edge_src_b = (edge_src.unsqueeze(0) + batch_offsets.unsqueeze(1)).reshape(-1)
        edge_dst_b = (edge_dst.unsqueeze(0) + batch_offsets.unsqueeze(1)).reshape(-1)

        log_ratio = (
            torch.log1p(flat_raw_armies[edge_src_b]) -
            torch.log1p(flat_raw_armies[edge_dst_b])
        )

        edge_attr = torch.stack([
            flat_armies[edge_src_b],
            flat_armies[edge_dst_b],
            flat_armies[edge_src_b] - flat_armies[edge_dst_b],
            log_ratio,
            (flat_owners[edge_src_b] == flat_owners[edge_dst_b]).float()
        ], dim=-1)

        x_flat = node_features.view(-1, node_features.shape[-1])

        batched_edge_index = torch.stack([edge_src_b, edge_dst_b], dim=0)

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

        # FiLM application
        if not self.is_likely_critic:
            gamma = 0.1*torch.tanh(self.phase_gamma(phase_emb))
            beta = self.phase_beta(phase_emb)
            h_global = h_global * (1.0 + gamma) + beta

        h_res = self.residual_block(h_global)
        h_res = self.norm_logits(h_res)

        full_logits = torch.full((batch_size, self.output_shape), float('-inf'), device=device)
        unique_phases = action_phase.unique()

        if not self.is_likely_critic:
            for phase in unique_phases:
                phase_mask = (action_phase == phase)
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

        torch.nn.init.normal_(self.phase_embedder.weight, mean=0.0, std=0.05)
        torch.nn.init.normal_(self.continent_embedder.weight, mean=0.0, std=0.05)

        if not self.is_likely_critic:
            torch.nn.init.orthogonal_(self.phase_gamma.weight, gain=0.05)
            torch.nn.init.zeros_(self.phase_gamma.bias)
            torch.nn.init.orthogonal_(self.phase_beta.weight, gain=0.05)
            torch.nn.init.zeros_(self.phase_beta.bias)
