import torch.nn.functional as F
from torch import Tensor, nn


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
