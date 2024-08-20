from pathlib import Path
from dataclasses import field, dataclass

from ..config import data_dir, tokenizer_dir


@dataclass
class TokenizerConfig:
    vocab_size: int = 1000
    special_tokens: set[str] = field(default_factory=lambda: {"<|endoftext|>"})
    train_text_file: Path = data_dir / "fineweb-tokenizer-train.txt"
    save_path: Path = tokenizer_dir / "tokenizer.bpe"
