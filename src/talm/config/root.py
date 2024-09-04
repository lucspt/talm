from pathlib import Path
from dataclasses import dataclass


@dataclass
class RootConfig:
    dirname: Path = Path(__file__).parent.parent  # repo/src/talm
    repo_root: Path = dirname.parent.parent  # /repo

    data_dir: Path = repo_root / "data"
    """Data download directory"""

    model_dir: Path = repo_root / "models"
    """Model checkpoints directory"""

    log_dir: Path = repo_root / "logs"
    """The directory where model logs will be stored"""

    tokenizer_dir: Path = repo_root / "tokenizers"
    """The directory where tokenizers will be stored"""
