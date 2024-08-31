from torch import Tensor, nn

from .mlp import MLP
from .attention import CausalSelfAttention


class DecoderBlock(nn.Module):
    """A transformer decoder block."""

    def __init__(
        self, n_head: int, n_embd: int, dropout: float = 0.0, ln_eps: float = 1e-5
    ) -> None:
        """Initialize a transformer decoder block.

        Args:
            n_head (int): The number of attention heads.
            n_embd (int): The number of embedding dims.
                Attention head size will be n_head / n_embd.
            dropout (float): A dropout probability factor. Defaults to 0.0
            ln_eps (float): Layer normalization epsilon value. Defaults to 1e-5.
        """
        super().__init__()
        self.multi_head_att = CausalSelfAttention(n_head=n_head, n_embd=n_embd)
        self.ln_1 = nn.LayerNorm(n_embd, eps=ln_eps)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout)
        self.ln_2 = nn.LayerNorm(n_embd, eps=ln_eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.multi_head_att(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
