import torch.nn.functional as F
from torch import Tensor, nn

from .modules import RMSNorm, DecoderBlock


class Model(nn.Module):
    """A decoder-only transformer base model for performing language modeling tasks"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_block: int,
        vocab_size: int,
        context_len: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        """Initialize the model

        Args:
            n_embd (int): The number of embedding dimensions.
            n_head (int): The number of heads to create the transformer `DecoderBlock`s with.
            n_block (int): The number of transformer decoder blocks to create.
            vocab_size (int): The vocab_size of the model. The number of possible token ids the model will output.
            context_len (int): The number of tokens a model can tend to during any forward pass.
            dropout (float): The amount of dropout to apply to the model, defaults to `0.0`.
            norm_eps (float): This value will be given as the `eps` argument to all `RMSNorm` layers within the model.
        """
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, n_embd)
        self.decoder = nn.Sequential(
            *(
                DecoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=dropout,
                    norm_eps=norm_eps,
                    max_seq_len=context_len * 2,
                )
                for _ in range(n_block)
            )
        )
        self.norm = RMSNorm(n_embd, eps=norm_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def compute_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """Computes the cross entropy loss between `logits` and `targets`"""
        B, T, C = logits.size()
        logits = logits.view(B * T, C)
        loss: Tensor = F.cross_entropy(logits, target.view(B * T))
        return loss

    def forward(self, x: Tensor) -> Tensor:
        x = self.tok_embedding(x)
        x = self.decoder(x)
        x = self.norm(x)
        logits: Tensor = self.lm_head(x)
        return logits
