import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """Implements Rotary Positional Embeddings

    See here: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        """Initialize a `RotaryPositionalEmbedding` module.

        Args:
            dim (int): The expected input shape of the embeddings
            max_seq_len (int): The maximum sequence length to be encoded.
            base (float): The base used when computing the rotation angles.
        """
        if dim % 2 != 0:
            raise ValueError("`dim` argument must be divisible by two")
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.base = base
        self._init_cache()
        self.cache: Tensor

    def _init_cache(self) -> None:
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        m = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(m, theta)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        cache = torch.stack([freqs_complex.real, freqs_complex.imag], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # not using kv-cache, no need for start_pos argument like in other implementations
        shape = x.shape
        seqlen = shape[1]
        rope_cache = self.cache[:seqlen]

        xshaped = x.reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out = x_out.flatten(3).type_as(x)
        return x_out


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
    """Implements causal self attention"""

    def __init__(
        self, n_head: int, n_embd: int, max_seq_len: int, dropout: float = 0.0
    ):
        """Initialize a `CausalSelfAttention` module.

        Args:
            n_head (int): The number of attention heads.
            n_embd (int): The number of embedding dims. Will be split by `n_head`.
            max_seq_len (int): Will be passed to `RotaryPositionalEmbedding`.
            dropout (float): A dropout probability rate.
        """
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
        self.head_dim = n_embd // n_head
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.size()
        n_head, head_dim = self.n_head, self.head_dim
        c_attn_out = self.c_attn(x)
        q: Tensor
        k: Tensor
        v: Tensor
        q, k, v = c_attn_out.split(self.n_embd, dim=2)  # (B, T, C), split on the C dim

        q = q.view(b, t, n_head, head_dim)
        k = k.view(b, t, n_head, head_dim)
        v = v.view(b, t, n_head, head_dim).transpose(1, 2)

        q, k = self.rope(q).transpose(1, 2), self.rope(k).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        # (B, n_head, T, head size) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(b, t, c)

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


class DecoderLayer(nn.Module):
    """A transformer decoder block."""

    def __init__(
        self,
        n_head: int,
        n_embd: int,
        max_seq_len: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        """Initialize a transformer decoder block.

        Args:
            n_head (int): The number of attention heads.
            n_embd (int): The number of embedding dims.
                Attention head size will be `n_embd // n_head`.
            dropout (float): A dropout probability factor. Defaults to 0.0
            norm_eps (float): Layer normalization epsilon value. Defaults to 1e-6.
        """
        super().__init__()
        self.attn = CausalSelfAttention(
            n_head=n_head, n_embd=n_embd, max_seq_len=max_seq_len
        )
        self.att_norm = RMSNorm(n_embd, eps=norm_eps)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout)
        self.mlp_norm = RMSNorm(n_embd, eps=norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.att_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
