import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Positional encoding that creates embedding tables
    for both positional and informational encoding.
    """

    def __init__(self, n_embd: int, vocab_size: int, context_len: int):
        """Initialize a PositionalEncoding.

        Args:
            n_embd (int): The number of embedding dims for the embedding tables.
            vocab_size (int): The total number of vocab tokens to create
                a token embedding table with
            context_len (int): The length of the context / number of positions to encode
                through a position embedding table.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_len, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        tx = self.token_embedding_table(x)  # (B, seq_len, n_embd)
        posx = self.position_embedding_table(
            torch.arange(x.size(1), device=x.device)
        )  # (B, n_embd)
        out: Tensor = tx + posx
        return out


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


class CausalSelfAttention(nn.Module):
    """A causal self attention layer for creating transformer decoder blocks

    Attributes:
        n_head (int): The number of attention heads to perform multi-headed attention with.
        n_embd (int): The number of embedding dims for each attention head (head size).
    """

    def __init__(self, n_head: int, n_embd: int, dropout: float = 0.0):
        if n_embd % n_head != 0:
            raise ValueError(
                "The number of embedding dims, specified by `n_embd`, "
                "must be divisible by the number of attention heads, specified with `n_head`. "
                f"Got {n_embd=}, {n_head=}"
            )
        super().__init__()

        self.n_head = n_head

        self.c_attn = nn.Linear(
            n_embd, 3 * n_embd, bias=False
        )  # 3 for query, key and value
        self.dropout_p = dropout
        self.o_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_embd = n_embd

    def create_heads(self, x: Tensor, B: int, T: int, C: int, n_head: int) -> Tensor:
        """Views and transposes a tensor of shape `(B, T, C)` to shape `(B, n_head, T, C // n_head)`.

        Args:
            x (Tensor): The tensor to transform.
            B (int): The batch of the tensor.
            T (int): The time dimension of the tensor.
            C (int): The channel dimension of the tensor.
            n_head (int): The desired number of attention heads.

        Returns:
            The transformed tensor.
        """
        return x.view(B, T, n_head, C // n_head).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        n_head = self.n_head
        c_attn_out = self.c_attn(x)
        q: Tensor
        k: Tensor
        v: Tensor
        q, k, v = c_attn_out.split(self.n_embd, dim=2)  # (B, T, C), split on the C dim

        q = self.create_heads(q, B, T, C, n_head)
        k = self.create_heads(k, B, T, C, n_head)
        v = self.create_heads(v, B, T, C, n_head)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        # (B, n_head, T, head size) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.o_dropout(self.proj(out))
        return out


class MLP(nn.Module):
    """Feed forward network / multi-layer perceptron."""

    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4, bias=False),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


class DecoderBlock(nn.Module):
    """A transformer decoder block."""

    def __init__(
        self, n_head: int, n_embd: int, dropout: float = 0.0, norm_eps: float = 1e-6
    ) -> None:
        """Initialize a transformer decoder block.

        Args:
            n_head (int): The number of attention heads.
            n_embd (int): The number of embedding dims.
                Attention head size will be n_head / n_embd.
            dropout (float): A dropout probability factor. Defaults to 0.0
            norm_eps (float): Layer normalization epsilon value. Defaults to 1e-6.
        """
        super().__init__()
        self.multi_head_att = CausalSelfAttention(n_head=n_head, n_embd=n_embd)
        self.att_norm = RMSNorm(n_embd, eps=norm_eps)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout)
        self.mlp_norm = RMSNorm(n_embd, eps=norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.multi_head_att(self.att_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
