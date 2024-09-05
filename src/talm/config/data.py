from pathlib import Path
from dataclasses import dataclass

from .root import RootConfig


@dataclass
class DataConfig:
    dataset_shard_size: int = int(1e8)
    """The size of a single dataset shard / file, in tokens"""

    data_dir: Path = RootConfig.data_dir
    """The directory to store data files"""

    fineweb_dataset_name: str = "HuggingFaceFW/fineweb"
    default_fineweb_dataset_sample: str = "sample-10BT"

    smol_dataset_name: str = "HuggingFaceTB/smollm-corpus"
    default_smol_dataset_sample: str = "cosmopedia-v2"

    sft_dataset_name: str = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
