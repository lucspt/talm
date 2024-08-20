from pathlib import Path
from dataclasses import field, dataclass

from .root import RootConfig


@dataclass
class TokenizerConfig:
    vocab_size: int = 1000
    special_tokens: set[str] = field(default_factory=lambda: {"<|endoftext|>"})
    train_text_file: Path = RootConfig.data_dir / "fineweb-tokenizer-train.txt"
    save_path: Path = RootConfig.tokenizer_path
