from dataclasses import field, dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 16
    n_epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = field(default_factory=lambda: (0.9, 0.95))
    adam_eps: float = 1e-8
