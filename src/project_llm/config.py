from pathlib import Path
from dataclasses import dataclass

dirname = Path(__file__).parent

data_dir = dirname / "_data"


@dataclass
class Config:
    tokenizer_path: str = str(dirname / "tokenizer.bpe")
    dataset_dir: Path = data_dir / "fineweb"
    dataset_shard_size: int = int(1e8)
