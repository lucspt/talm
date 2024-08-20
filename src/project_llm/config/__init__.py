from dataclasses import dataclass

from .data import DataConfig
from .root import RootConfig
from .model import ModelConfig
from .training import TrainingConfig


# we intentionally don't add tokenizer config, it's not needed here.
@dataclass
class Config(RootConfig, ModelConfig, DataConfig, TrainingConfig): ...
