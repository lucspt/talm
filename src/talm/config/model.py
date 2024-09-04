from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_embd: int = 64
    n_head: int = 4
    context_len: int = 16
    dropout: float = 0.2
    norm_eps: float = 1e-6
    n_transformer_blocks: int = 4
