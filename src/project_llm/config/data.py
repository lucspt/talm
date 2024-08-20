from pathlib import Path
from dataclasses import dataclass

from .root import RootConfig


@dataclass
class DataConfig:
    dataset_shard_size: int = int(1e8)
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_sample: str = "sample-10BT"
    dataset_dir: Path = RootConfig.data_dir / "fineweb"
