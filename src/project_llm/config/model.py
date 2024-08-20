from dataclasses import field, dataclass


@dataclass
class ModelConfig:
    n_embd: int = 64
    n_head: int = 4
    context_len: int = 16
    dropout: float = 0.2
    ln_eps: float = 1e-5
    lr: float = 3e-4
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = field(default_factory=lambda: (0.9, 0.95))
    adam_eps: float = 1e-8
    batch_size: int = 16
    n_epochs: int = 1
