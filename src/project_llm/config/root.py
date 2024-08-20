from pathlib import Path
from dataclasses import dataclass


@dataclass
class RootConfig:
    dirname: Path = Path(__file__).parent
    repo_root: Path = dirname.parent.parent
    data_dir: Path = repo_root / "data"
    model_dir: Path = repo_root / "models"
    log_dir: Path = repo_root / "logs"
    tokenizer_dir: Path = repo_root / "tokenizers"
    tokenizer_path: Path = tokenizer_dir / "tokenizer.bpe"
