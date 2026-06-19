import torch
from torch import nn
from torch_geometric.nn import TransformerConv


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