import torch.nn.functional as F
from torch import Tensor, nn

from .modules import RMSNorm, DecoderLayer
from ..config.model import ModelConfig


class Model(nn.Module):
    """A decoder-only transformer model for performing language modeling tasks


    Attributes:
        config (ModelConfig): The configuration of the model
    """

    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        """Initialize the model

        Args:
            config (ModelConfig): The model configuration dataclass.
            vocab_size (int): The model's desired vocab size
        """
        super().__init__()
        self.config = config
        self.tok_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.decoder = nn.Sequential(
            *(
                DecoderLayer(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    dropout=config.dropout,
                    norm_eps=config.norm_eps,
                    max_seq_len=config.context_len * 2,
                )
                for _ in range(config.n_transformer_layers)
            )
        )
        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

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
