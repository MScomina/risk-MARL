import torch
from torch import nn
from torch_geometric.nn import TransformerConv

from networkx import DiGraph

from .normalization import PureLayerNorm, RunningMeanStd
from environment.risk_utils import RiskPhase


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


class FiLMBlock(nn.Module):
    # https://arxiv.org/pdf/1709.07871 (FiLM: Visual Reasoning with a General Conditioning Layer)
    def __init__(self, input_shape : int, output_shape : int, gamma_scaling : float = 0.1):
        super().__init__()
        self.gamma = nn.Linear(in_features=input_shape, out_features=output_shape)
        self.beta = nn.Linear(in_features=input_shape, out_features=output_shape)
        self.gamma_scaling = gamma_scaling

        self._init_weights()

    def forward(self, x, cond):
        gamma = self.gamma_scaling*torch.tanh(self.gamma(cond))
        beta = self.beta(cond)
        return x * (1.0 + gamma) + beta

    def _init_weights(self):
        torch.nn.init.orthogonal_(self.gamma.weight, gain=0.05)
        torch.nn.init.zeros_(self.gamma.bias)
        torch.nn.init.orthogonal_(self.beta.weight, gain=0.05)
        torch.nn.init.zeros_(self.beta.bias)


class NodeFeatureExtractor(nn.Module):

    def __init__(self,
                map_graph : DiGraph,
                n_players : int,
                embed_space_continent : int = 6,
                embed_space_phase : int = 16,
                 ):
        super().__init__()

        self.map_graph = map_graph
        self.n_players = n_players

        self.continent_dim = embed_space_continent
        self.embed_space_phase = embed_space_phase
        self.phase_embedder = nn.Embedding(
            num_embeddings=len(RiskPhase), 
            embedding_dim=self.embed_space_phase
        )
        unique_continents = sorted(set([
            self.map_graph.nodes[node].get('continent', 'Unknown')
            for node in self.map_graph.nodes
        ]))
        self.continent_to_idx = {name: i for i, name in enumerate(unique_continents)}
        self.continent_embedder = nn.Embedding(20, self.continent_dim)

        self.num_owner_categories = 3

        self.army_norm = RunningMeanStd(1)
        self.owner_army_norm = RunningMeanStd(1)
        self.owner_territory_norm = RunningMeanStd(1)

        self._init_weights()

    def get_num_parameters(self) -> int:
        return (
            self.num_owner_categories +
            1 + # Normalized armies
            1 + # Owner army total
            1 + # Owner territory total
            self.embed_space_phase + 
            self.continent_dim +
            1 + # Node selected one-hot
            1 # Edge selected indicator
        )

    def forward(self, obs_clean : tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:

        device = next(self.parameters()).device

        territory_owner, selected_node, selected_edge, action_phase, armies_tensor = obs_clean

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

        return torch.cat(features_array, dim=2), normalized_armies

    def _init_weights(self):
        torch.nn.init.normal_(self.phase_embedder.weight, mean=0.0, std=0.05)
        torch.nn.init.normal_(self.continent_embedder.weight, mean=0.0, std=0.05)

class EdgeFeatureExtractor(nn.Module):

    def __init__(self,
                map_graph : DiGraph):
        
        super().__init__()
        self.map_graph = map_graph

    def forward(self, obs_clean : tuple[torch.Tensor, ...], normalized_armies : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        device = next(self.parameters()).device

        territory_owner, selected_node, selected_edge, action_phase, armies_tensor = obs_clean

        int_edges = [(self.map_graph.graph["node_to_idx"][u], self.map_graph.graph["node_to_idx"][v])
                     for u, v in self.map_graph.edges]
        edge_index = torch.tensor(list(int_edges), dtype=torch.long, device=device).t().contiguous()

        batch_size = territory_owner.shape[0]
        num_nodes = territory_owner.shape[1]

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

        batched_edge_index = torch.stack([edge_src_b, edge_dst_b], dim=0)

        return batched_edge_index, edge_attr