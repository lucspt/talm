from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn

from .block import DecoderBlock
from .encoding import PositionalEncoding


class Model(nn.Module):
    """A decoder-only transformer base model for performing language modeling tasks"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        vocab_size: int,
        context_len: int,
        dropout: float = 0.0,
        ln_eps: float = 1e-5,
    ) -> None:
        """Initialize the model

        Args:
            n_embd (int): The number of embedding dimensions.
            n_head (int): The number of heads to create the transformer `DecoderBlocks` with.
            vocab_size (int): The vocab_size of the model. The number of possible token ids the model will output.
            context_len (int): The number of tokens a model can tend to during any forward pass.
            dropout (float): The amount of dropout to apply to the model, defaults to `0.0`.
            ln_eps (float): This value will be given as the `eps` argument to all `nn.LayerNorm` layers within the model.
        """
        super().__init__()
        self.positional_encoding = PositionalEncoding(
            n_embd=n_embd, vocab_size=vocab_size, context_len=context_len
        )
        self.decoder = nn.Sequential(
            *(
                DecoderBlock(
                    n_embd=n_embd, n_head=n_head, dropout=dropout, ln_eps=ln_eps
                )
                for _ in range(n_head)
            )
        )
        self.ln_f = nn.LayerNorm(n_embd, eps=ln_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _compute_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """Computes the cross entropy loss between `logits` and `targets`"""
        B, T, C = logits.size()
        logits = logits.view(B * T, C)
        return F.cross_entropy(logits, target.view(B * T))

    def forward(
        self, x: Tensor, target: Optional[Tensor] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        x = self.positional_encoding(x)
        x = self.decoder(x)
        x = self.ln_f(x)
        logits: Tensor = self.lm_head(x)

        if target is not None:
            loss = self._compute_loss(logits=logits, target=target)
            return logits, loss
        else:
            return logits, None
