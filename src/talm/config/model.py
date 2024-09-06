from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_embd: int = 64
    """The number of embedding dimensions"""

    n_head: int = 8
    """The number of attention heads"""

    context_len: int = 128
    """The maximum sequence length the model will be able to handle"""

    dropout: float = 0.2
    """The dropout probability factor"""

    norm_eps: float = 1e-6
    """The epsilon value for all RMSNorm modules"""

    n_transformer_layers: int = 4
    """The number of transformer decoder layers"""
