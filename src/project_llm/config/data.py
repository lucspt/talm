from pathlib import Path
from dataclasses import dataclass

from .root import RootConfig


@dataclass
class DataConfig:
    dataset_shard_size: int = int(1e8)
    fineweb_dataset_name: str = "HuggingFaceFW/fineweb"
    default_fineweb_dataset_sample: str = "sample-10BT"
    data_dir: Path = RootConfig.data_dir
    smol_dataset_name: str = "HuggingFaceTB/smollm-corpus"
    default_smol_dataset_sample: str = "cosmopedia-v2"
