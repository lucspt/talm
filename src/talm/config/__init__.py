from dataclasses import dataclass

from .data import DataConfig
from .root import RootConfig
from .model import ModelConfig
from .training import TrainingConfig

__all__ = ["ModelConfig", "DataConfig", "RootConfig", "TrainingConfig", "Config"]


# we intentionally don't add tokenizer config, it's not needed here.
@dataclass
class Config(RootConfig, ModelConfig, DataConfig, TrainingConfig): ...
