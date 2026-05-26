# https://arxiv.org/pdf/1312.6120 (Exact solutions to the nonlinear dynamics of learning in deep linear neural networks)
# https://papers.nips.cc/paper_files/paper/2017/file/5d44ee6f2c3f71b73125876103c8f6c4-Paper.pdf (Self-Normalizing Neural Networks)

import torch
import torch.nn as nn
import numpy as np


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

    def __init__(self, input_shape : int, output_shape : int, hidden_dim : int, res_blocks : int = 6, activation_function : nn.Module = nn.ReLU):

        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_dim = hidden_dim

        self.activation_function = activation_function

        self.input_layer = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation_function()
        )

        self.res_block_stack = nn.Sequential(
            *[ResidualBlock(channels=hidden_dim, activation_fn=self.activation_function) for _ in range(res_blocks)]
        )

        self.qa_head = nn.Linear(hidden_dim, output_shape)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=torch.nn.init.calculate_gain(self.activation_function._get_name()))
                torch.nn.init.constant_(m.bias, 0)

        torch.nn.init.orthogonal_(self.qa_head.weight, gain=0.01)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.res_block_stack(x)
        logits = self.qa_head(x)
        
        return logits