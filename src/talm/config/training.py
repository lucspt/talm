from dataclasses import field, dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 16
    """Training batch size"""

    n_epochs: int = 1
    """Number of training epochs"""

    max_lr: float = 6e-4
    """The max learning rate"""

    min_lr: float = max_lr * 0.1
    """The minimum learning rate"""

    weight_decay: float = 0.01
    """AdamW weight decay"""

    adam_betas: tuple[float, float] = field(default_factory=lambda: (0.9, 0.95))
    """AdamW betas"""

    adam_eps: float = 1e-8
    """AdamW epsilon value"""

    gradient_clip_value: float = 1.0
    """Clip gradients with this value"""


@dataclass
class SFTConfig(TrainingConfig):
    n_epochs: int = 1
    max_lr: float = 2e-5
    min_lr: float = 5e-5
    dataset_name: str = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
