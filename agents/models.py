# https://arxiv.org/pdf/1312.6120 (Exact solutions to the nonlinear dynamics of learning in deep linear neural networks)
# https://papers.nips.cc/paper_files/paper/2017/file/5d44ee6f2c3f71b73125876103c8f6c4-Paper.pdf (Self-Normalizing Neural Networks)

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from environment.risk_utils import RiskPhase

def shape_obs(obs_space : spaces.Dict, is_embedded : bool, embed_space_owners : int | None = None, embed_space_nodes : int | None = None, embed_space_edges : int | None = None) -> int:

    input_size = 0
    # Embedding encoding for ownership of territories.
    input_size += obs_space["territory_owner"].shape[0] * (embed_space_owners if is_embedded else 1)
    input_size += obs_space["number_of_armies"].shape[0]
    # One hot encoding for action phases.
    input_size += len(RiskPhase)
    # Troops to place
    input_size += 1
    input_size += (embed_space_nodes if is_embedded else 1)
    input_size += (embed_space_edges if is_embedded else 1)

    if "cards_in_hand" in obs_space.keys():
        # Types of cards
        input_size += obs_space["cards_in_hand"].shape[0]
        # Amount cards opponent
        input_size += obs_space["amount_cards_others"].shape[0]


    return input_size

class ResidualBlock(nn.Module):

    def __init__(self, channels, activation_fn : nn.Module = nn.ReLU, starting_residual_scale : float = 0.1):
        super().__init__()
        self.activation = activation_fn()
        self.norm = nn.LayerNorm(channels)
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            self.activation
        )
        self.residual_scale = nn.Parameter(torch.tensor(starting_residual_scale))

    def forward(self, x):
        x_norm = self.norm(x)
        return self.activation(x_norm + self.residual_scale * self.block(x_norm))


class ResidualNetwork(nn.Module):

    def __init__(self, obs_space : spaces.Dict, action_space : spaces.Discrete | int, hidden_dim : int = 128, activation_function : nn.Module = nn.ReLU,
                 res_blocks : int = 6, embed_space_owners : int = 6, embed_space_nodes : int = 8, embed_space_edges : int = 16, map_decoder_depth : int = 3):

        super().__init__()
        obs_space = obs_space["observation"]
        self.input_shape = shape_obs(
            obs_space=obs_space, 
            is_embedded=True, 
            embed_space_owners=embed_space_owners,
            embed_space_edges=embed_space_edges,
            embed_space_nodes=embed_space_nodes
        )
        if isinstance(action_space, spaces.Discrete):
            self.output_shape = action_space.n
        else:
            self.output_shape = action_space

        self.hidden_dim = hidden_dim
        self.res_blocks = res_blocks
        self.embed_space_owners = embed_space_owners
        self.embed_space_nodes = embed_space_nodes
        self.embed_space_edges = embed_space_edges
        self.map_decoder_depth = map_decoder_depth

        if self.map_decoder_depth >= self.res_blocks:
            raise ValueError(f"Map decoder can't be bigger than the number of res_blocks: {self.map_decoder_depth} >= {self.res_blocks}")

        self.activation_function = activation_function

        self.map_decoder = nn.Sequential(
            *[ResidualBlock(channels=self.input_shape-len(RiskPhase), activation_fn=self.activation_function) for _ in range(self.map_decoder_depth)]
        )

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.activation_function()
        )

        self.res_block_stack = nn.Sequential(
            *[ResidualBlock(channels=self.hidden_dim, activation_fn=self.activation_function) for _ in range(self.res_blocks-self.map_decoder_depth)]
        )

        self.owner_embedder = nn.Embedding(
            num_embeddings=(obs_space["territory_owner"].high[0] - obs_space["territory_owner"].low[0])+1,
            embedding_dim=self.embed_space_owners
        )

        self.nodes_embedder = nn.Embedding(
            num_embeddings=(obs_space["selected_node"].n)+1,
            embedding_dim=self.embed_space_nodes
        )

        self.edges_embedder = nn.Embedding(
            num_embeddings=(obs_space["selected_edge"].n)+1,
            embedding_dim=self.embed_space_edges
        )

        self.qa_head = nn.Linear(self.hidden_dim, self.output_shape)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=torch.nn.init.calculate_gain(self.activation_function.__name__.lower()))
                torch.nn.init.constant_(m.bias, 0)

        torch.nn.init.orthogonal_(self.qa_head.weight, gain=0.01)


    def forward(self, obs, state=None, info={}):

        device = next(self.parameters()).device

        territory_owner = torch.as_tensor(obs["territory_owner"], dtype=torch.long, device=device)
        selected_node = torch.as_tensor(obs["selected_node"], dtype=torch.long, device=device)
        selected_edge = torch.as_tensor(obs["selected_edge"], dtype=torch.long, device=device)
        action_phase = torch.as_tensor(obs["action_phase"], dtype=torch.long, device=device)
        
        batch_size = territory_owner.shape[0]
        
        embedded_owners = self.owner_embedder(territory_owner + 1)
        embedded_owners = torch.flatten(embedded_owners, start_dim=1)

        embedded_node = torch.flatten(self.nodes_embedder(selected_node + 1), start_dim=1)
        embedded_edge = torch.flatten(self.edges_embedder(selected_edge + 1), start_dim=1)

        phase_one_hot = torch.nn.functional.one_hot(action_phase, num_classes=len(RiskPhase)).float()
        phase_one_hot = phase_one_hot.view(batch_size, -1)

        normalized_armies = torch.as_tensor(np.log1p(obs["number_of_armies"]), dtype=torch.float32, device=device)
        normalized_reinforcements = torch.as_tensor(np.log1p(obs["troops_to_place"]), dtype=torch.float32, device=device).view(-1, 1)

        if "cards_in_hand" in obs:
            cards_normalized = torch.as_tensor(np.log1p(obs["cards_in_hand"]), dtype=torch.float32, device=device)
            hands_normalized = torch.as_tensor(np.log1p(obs["amount_cards_others"]), dtype=torch.float32, device=device)
        else:
            cards_normalized = torch.empty((batch_size, 0), dtype=torch.float32, device=device)
            hands_normalized = torch.empty((batch_size, 0), dtype=torch.float32, device=device)

        map_data = torch.cat([
            embedded_owners,
            embedded_node,
            embedded_edge,
            normalized_armies,
            normalized_reinforcements,
            cards_normalized,
            hands_normalized
        ], dim=1)

        map_representation = self.map_decoder(map_data)

        mixed_representation = torch.cat([
            map_representation,
            phase_one_hot
        ], dim=1)

        mixed_representation = self.input_layer(mixed_representation)
        mixed_representation = self.res_block_stack(mixed_representation)
        logits = self.qa_head(mixed_representation)

        return logits, state