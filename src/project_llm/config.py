from pathlib import Path
from dataclasses import dataclass

dirname = Path(__file__).parent

repo_root = dirname.parent.parent

data_dir = repo_root / "data"

tokenizer_dir = repo_root / "tokenizer"

model_dir = repo_root / "models"


@dataclass
class Config:
    tokenizer_path: Path = tokenizer_dir / "tokenizer.bpe"
    model_dir: Path = model_dir
    dataset_dir: Path = data_dir / "fineweb"
    dataset_shard_size: int = int(1e8)
