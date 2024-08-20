from dataclasses import dataclass

from .data import DataConfig
from .root import RootConfig
from .model import ModelConfig


@dataclass
class Config(RootConfig, ModelConfig, DataConfig): ...
