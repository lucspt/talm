from pathlib import Path
from dataclasses import field, dataclass

from .root import RootConfig


@dataclass
class TokenizerConfig:
    special_tokens: set[str] = field(default_factory=lambda: {"<|endoftext|>"})
    tokenizer_train_file: Path = RootConfig.data_dir / "fineweb-tokenizer-train.txt"
    registry_dir: Path = RootConfig.tokenizer_dir
