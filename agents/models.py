# https://arxiv.org/pdf/1312.6120 (Exact solutions to the nonlinear dynamics of learning in deep linear neural networks)

from networkx import DiGraph
import torch
import torch_geometric as torchg
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from environment.risk_utils import RiskPhase
from tianshou.data.batch import Batch
from tianshou.algorithm.algorithm_base import Policy
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, TransformerConv, Sequential as PyGSequential

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

    def __init__(self, channels, activation_fn : nn.Module = nn.ReLU, starting_residual_scale : float = 0.1, is_residual : bool = True):
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
    
class GraphNetwork(nn.Module):

    def __init__(self, 
                obs_space : spaces.Dict, 
                action_space : spaces.Discrete | int, 
                map_graph : DiGraph, 
                gnn_hidden_size : int = 64,
                res_hidden_size : int = 128,
                activation_function : nn.Module = nn.ReLU,
                embed_space_owners : int = 4,
                embed_space_phase : int = 16,
                residual_depth : int = 3,
                starting_residual_scale : float = 0.1,
                n_heads : int = 8
                ):
        super().__init__()
        if "observation" in obs_space.keys():
            obs_space = obs_space["observation"]

        self.map_graph : DiGraph = map_graph
        int_edges = [(self.map_graph.graph["node_to_idx"][u], self.map_graph.graph["node_to_idx"][v]) for u, v in self.map_graph.edges]
        edge_index = torch.tensor(
            list(int_edges), dtype=torch.long
        ).t().contiguous()
        self.register_buffer("edge_index", edge_index)
        self.gnn_hidden_size = gnn_hidden_size
        self.res_hidden_size = res_hidden_size
        self.n_nodes = obs_space["selected_node"].n - 1
        self.n_edges = obs_space["selected_edge"].n - 1
        self.activation_function = activation_function
        self.register_buffer(
            "max_log_armies", 
            torch.log1p(torch.tensor(obs_space["number_of_armies"].high[0] - obs_space["number_of_armies"].low[0], dtype=torch.float32))
        )
        if isinstance(action_space, spaces.Discrete):
            self.output_shape = action_space.n
        else:
            self.output_shape = action_space
        self.embed_space_owners = embed_space_owners
        self.embed_space_phase = embed_space_phase
        self.residual_depth = residual_depth
        self.starting_residual_scale = starting_residual_scale
        self.n_heads = n_heads
        self.is_likely_critic = (self.output_shape == 1)

        f_in = 1  + self.embed_space_owners + self.embed_space_phase + (0 if self.is_likely_critic else 1+1)    # n_armies, selected_node, selected_edge, territory_owner, phase info

        self.owner_embedder = nn.Embedding(
            num_embeddings=(obs_space["territory_owner"].high[0] - obs_space["territory_owner"].low[0]+1),
            embedding_dim=self.embed_space_owners
        )

        self.phase_embedder = nn.Embedding(
            num_embeddings=len(RiskPhase), 
            embedding_dim=self.embed_space_phase
        )

        self.gnn = PyGSequential('x, edge_index, edge_attr', [
            (TransformerConv(f_in, self.gnn_hidden_size, heads=self.n_heads, concat=True, dropout=0.1, edge_dim=4), 'x, edge_index, edge_attr -> x'),
            nn.Linear(self.n_heads*self.gnn_hidden_size, self.gnn_hidden_size),
            PureLayerNorm(self.gnn_hidden_size),
            self.activation_function(),
            
            (TransformerConv(self.gnn_hidden_size, self.gnn_hidden_size, heads=self.n_heads, concat=True, dropout=0.1, edge_dim=4), 'x, edge_index, edge_attr -> x'),
            nn.Linear(self.n_heads*self.gnn_hidden_size, self.gnn_hidden_size),
            PureLayerNorm(self.gnn_hidden_size),
            self.activation_function()
        ])

        self.mean_pool = torchg.nn.global_mean_pool
        self.max_pool = torchg.nn.global_max_pool

        self.residual_block = nn.Sequential(
            *[ResidualBlock(
                channels=self.res_hidden_size,
                activation_fn=self.activation_function,
                starting_residual_scale=self.starting_residual_scale
            ) for _ in range(self.residual_depth)]
        )

        projection_size = (2*self.gnn_hidden_size) + self.embed_space_phase + 1 # mean+max pool, embed phase and reinforcements

        if "cards_in_hand" in obs_space.keys():
            projection_size += obs_space["cards_in_hand"].shape[0] + obs_space["amount_cards_others"].shape[0]
            
        self.global_projection = nn.Linear(projection_size, self.res_hidden_size)

        self.qa_heads = nn.ModuleList(
            [nn.Linear(self.res_hidden_size, self.output_shape) for _ in range(len(RiskPhase))]
        )

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

        batch_size = territory_owner.shape[0]

        phase_emb = self.phase_embedder(action_phase)
        phase_node_context = phase_emb.unsqueeze(1).repeat(1, self.n_nodes, 1)

        owner_embedding = self.owner_embedder(territory_owner+1)        
        armies_tensor = torch.as_tensor(obs_dict["number_of_armies"], dtype=torch.float32, device=device)
        normalized_armies = torch.log1p(armies_tensor) / self.max_log_armies
        normalized_armies = normalized_armies.unsqueeze(2)

        if not self.is_likely_critic:
            node_select_onehot = torch.zeros(batch_size, self.n_nodes, device=device)
            valid_nodes = selected_node >= 0
            node_select_onehot[valid_nodes, selected_node[valid_nodes]] = 1.0
            node_select_onehot = node_select_onehot.unsqueeze(2)

            edge_node_indicator = torch.zeros(batch_size, self.n_nodes, device=device)
            valid_edges = selected_edge >= 0

            if valid_edges.any():
                active_edges = selected_edge[valid_edges]
                src_nodes = self.edge_index[0, active_edges]
                tgt_nodes = self.edge_index[1, active_edges]

                batch_indices = torch.arange(batch_size, device=device)[valid_edges]
                edge_node_indicator[batch_indices, src_nodes] = -1.0
                edge_node_indicator[batch_indices, tgt_nodes] = 1.0

            edge_node_indicator = edge_node_indicator.unsqueeze(2)

        reinforce_tensor = torch.as_tensor(obs_dict["troops_to_place"], dtype=torch.float32, device=device)
        normalized_reinforcements = torch.log1p(reinforce_tensor).view(-1, 1) / self.max_log_armies

        features_array = [
            owner_embedding,
            normalized_armies,
            phase_node_context
        ]

        if not self.is_likely_critic:
            features_array.extend([node_select_onehot, edge_node_indicator])

        # Computing node and edge features for GNN
        node_features = torch.cat(features_array, dim=2)

        node_armies = normalized_armies.squeeze(-1)
        node_owners = territory_owner
        edge_src, edge_dst = self.edge_index

        batch_offsets = torch.arange(batch_size, device=device) * self.n_nodes

        edge_src_b = edge_src.unsqueeze(0) + batch_offsets.unsqueeze(1)
        edge_dst_b = edge_dst.unsqueeze(0) + batch_offsets.unsqueeze(1)

        edge_src_b = edge_src_b.reshape(-1)
        edge_dst_b = edge_dst_b.reshape(-1)

        batched_edge_index = torch.stack([edge_src_b, edge_dst_b], dim=0)

        flat_armies = node_armies.view(batch_size * self.n_nodes)
        flat_owners = node_owners.view(batch_size * self.n_nodes)

        edge_attr = torch.stack([
            flat_armies[edge_src_b],
            flat_armies[edge_dst_b],
            flat_armies[edge_src_b] - flat_armies[edge_dst_b],
            (flat_owners[edge_src_b] == flat_owners[edge_dst_b]).float()
        ], dim=-1)

        x_flat = node_features.view(batch_size * self.n_nodes, -1)

        h_nodes = self.gnn(x_flat, batched_edge_index, edge_attr)

        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(self.n_nodes)
        h_mean_pooled = self.mean_pool(h_nodes, batch_idx)
        h_max_pooled = self.max_pool(h_nodes, batch_idx)

        if "cards_in_hand" in obs_dict:
            cards_normalized = torch.log1p(torch.as_tensor(obs_dict["cards_in_hand"], dtype=torch.float32, device=device))
            hands_normalized = torch.log1p(torch.as_tensor(obs_dict["amount_cards_others"], dtype=torch.float32, device=device))
        else:
            cards_normalized = torch.zeros((batch_size, 0), dtype=torch.float32, device=device)
            hands_normalized = torch.zeros((batch_size, 0), dtype=torch.float32, device=device)

        h_global = torch.cat([
            h_mean_pooled,
            h_max_pooled, 
            phase_emb,
            normalized_reinforcements,
            cards_normalized,
            hands_normalized
        ], dim=-1)

        h_global = self.activation_function()(self.global_projection(h_global))

        h_res = self.residual_block(h_global)
        logits = torch.zeros(batch_size, self.output_shape, device=device)

        for phase_idx, head in enumerate(self.qa_heads):
            mask = (action_phase == phase_idx)
            if mask.any():
                logits[mask] = head(h_res[mask])
        
        if "mask" in obs and self.output_shape != 1:
            mask = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device)
            logits = torch.where(mask, logits, torch.tensor(-1e8, device=device))

        if self.is_likely_critic:
            return logits.cpu()

        return logits.cpu(), state


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

        for head in self.qa_heads:
            torch.nn.init.orthogonal_(head.weight, gain=0.01)
            torch.nn.init.constant_(head.bias, 0)

        torch.nn.init.normal_(self.owner_embedder.weight, mean=0.0, std=0.05)
        torch.nn.init.normal_(self.phase_embedder.weight, mean=0.0, std=0.05)
