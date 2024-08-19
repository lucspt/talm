from pathlib import Path
from dataclasses import dataclass

from ..config import Config

config = Config()


@dataclass
class DataConfig:
    shard_size: int = int(1e8)
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_sample: str = "sample-10BT"
    save_dir: Path = config.dataset_dir
    tokenizer_path: str = config.tokenizer_path
