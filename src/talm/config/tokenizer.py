from pathlib import Path
from dataclasses import field, dataclass

from .root import RootConfig


@dataclass
class TokenizerConfig:
    special_tokens: set[str] = field(
        default_factory=lambda: {
            "<|endoftext|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
        }
    )
    tokenizer_train_file: Path = RootConfig.repo_root / "tiny-shakespeare.txt"
    registry_dir: Path = RootConfig.tokenizer_dir
