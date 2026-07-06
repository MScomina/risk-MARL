# https://link.springer.com/content/pdf/10.1007/s10994-025-06822-0.pdf (Analyzing the effect of residual connections to oversmoothing in graph neural networks)

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class RunningMeanStd(nn.Module):
    '''
        Running mean/std module, it computes the mean and standard deviation of
        the variables progressively to get more accurate over time.
        This assumes there are no wildly changing values, so this does
        hurt the generalization capabilities of networks.
        It does nonetheless work very well for static maps.
        It might need revision if one wants to improve the agent's generalization on unseen maps.
    '''
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
    '''
        Just a normal LayerNorm implementation.
        For some reason on ROCm the normal LayerNorm implementation generates NaN values on high batch sizes.
        If LayerNorm works properly with your setup, just replace this with the Torch version, as it is probably faster.
    '''

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
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
    '''
        This torch module consists in applying the FiLM technique implemented in the paper
        "FiLM: Visual Reasoning with a General Conditioning Layer". 
        It consists in having the network learn a set of mean/std values to condition the
        values of the internal representation with, based on a conditioning tensor.
        This approach seems to be more effective than just concatenating the tensor.
        Paper: https://arxiv.org/pdf/1709.07871
    '''
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


class PolicyHead(nn.Module):
    '''
        Policy head used in the GraphNetwork.
        It's just a bunch of residual blocks outputting a single logit for each node/edge it's run on.
    '''
    def __init__(self, feature_dim : int, res_depth : int = 3, activation_fn : nn.Module = nn.ReLU, starting_residual_scale : float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.activation_fn = activation_fn
        self.res_depth = res_depth
        self.starting_residual_scale = starting_residual_scale
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(
                channels=self.feature_dim,
                activation_fn=self.activation_fn,
                starting_residual_scale=self.starting_residual_scale
            ) for _ in range(self.res_depth)]
        )
        self.output_layer = nn.Linear(self.feature_dim, 1)

        self._init_weights()

    def forward(self, x):
        return self.output_layer(x)

    def _init_weights(self):

        for m in self.res_blocks.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

        for m in self.output_layer.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=0.01)
                torch.nn.init.zeros_(m.bias)
        
class ResidualBlock(nn.Module):
    '''
        A residual block, consisting of a linear MLP that can be skipped through a residual connection.
        Residual connections have been in use ever since the paper "Deep Residual Learning for Image Recognition"
        has been released. They allow for better gradient flow and allow for layers to deactivate themselves without
        impeding the network function as a whole.
        Paper: https://arxiv.org/abs/1512.03385
    '''

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
    '''
        An equivalent to the ResidualBlock, just for the GNN case.
        It has a residual connection that lets gradients flow better and lets layers deactivate themselves if not needed.
        They apparently are even more effective on GNNs, as they reduce the effect of oversmoothing slightly as well.
        (GNNs that are too deep tend to converge to a uniform representation, they look the same everywhere)
        Papers: 
            - https://arxiv.org/abs/1512.03385 (Residual connections)
            - https://arxiv.org/pdf/1706.02216 (GraphSAGE)
            - https://link.springer.com/content/pdf/10.1007/s10994-025-06822-0.pdf (Oversmoothing)
    '''

    def __init__(
        self,
        in_channels,
        hidden_size,
        activation_fn=nn.ReLU,
        edge_dim=5,
        is_residual=True,
        starting_residual_scale=1.0
    ):
        super().__init__()

        self.is_residual = is_residual

        self.conv = SAGEConv(
            in_channels=in_channels,
            out_channels=hidden_size,
            normalize=True
        )

        if self.is_residual and in_channels != hidden_size:
            self.skip = nn.Linear(in_channels, hidden_size)
        else:
            self.skip = None

        self.norm = PureLayerNorm(hidden_size)
        self.activation = activation_fn()
        if self.is_residual:
            self.residual_scale = nn.Parameter(torch.tensor(starting_residual_scale))

    def forward(self, x, edge_index):

        if self.is_residual:
            residual = x
            if self.skip is not None:
                residual = self.skip(residual)

        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.activation(x)

        if self.is_residual:
            x = (self.residual_scale * x) + residual

        return x