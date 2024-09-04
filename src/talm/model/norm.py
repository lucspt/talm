import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Implements root mean square layer normalization.

    See here: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize an RMSNorm module.

        Args:
            dim (int): The input size expected.
            eps (float): The value that will be added to avoid a division by zero when rescaling.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * self._norm(x.float()).type_as(x)
