from pathlib import Path
from dataclasses import dataclass

from ..config import Config

root_config = Config()


@dataclass
class DataConfig:
    shard_size: int = int(1e8)
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_sample: str = "sample-10BT"
    save_dir: Path = root_config.dataset_dir
    tokenizer_path: str = root_config.tokenizer_path
