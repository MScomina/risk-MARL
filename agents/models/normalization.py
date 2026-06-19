import torch
import torch.nn as nn


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