import os
import sys
from typing import Literal, Optional
from pathlib import Path
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm  # type: ignore
from datasets import DatasetDict, load_dataset  # type: ignore
from tokencoder import Tokenizer
from numpy.typing import NDArray, ArrayLike

from ..logger import create_logger
from .helpers import is_folder_empty
from ..config.data import DataConfig

DatasetName = Literal["fineweb", "smol"]
ALLOWED_DATASETS: tuple[DatasetName, ...] = ("fineweb", "smol")
DS_SAMPLES = {
    "fineweb": {"sample-10BT", "sample-100BT", "sample-350BT"},
    "smol": {"python-edu", "fineweb-edu", "cosmopedia-v2"},
}

config = DataConfig()

logger = create_logger(__name__)


def abort_if_dataset_dir_not_empty(dataset_dir: Path) -> None:
    if dataset_dir.exists() and not is_folder_empty(dataset_dir):
        logger.error(
            f"Dataset dir '{dataset_dir}' already exists, and is not empty. "
            "Aborting to not overwrite any files"
        )
        sys.exit(1)


def update_progress_bar(
    bar: None | tqdm,
    n: int | float,
    total: Optional[int] = None,
    nth_shard: Optional[int] = None,
) -> tqdm:
    if bar is None:
        s = nth_shard + 1 if nth_shard == 0 else nth_shard
        bar = tqdm(
            total=total,
            desc=f"Shard {s} ({"val" if nth_shard == 0 else "train"})",
        )
    bar.update(n)
    return bar


def write_file(dir: Path, tokens: ArrayLike, shard: int) -> Path:
    split_prefix = "val" if shard == 0 else "train"
    shard = 1 if split_prefix == "val" else shard
    pth = dir / f"{split_prefix}_{shard:06d}"
    np.save(pth, tokens)
    return pth


def tokenize(ds_example: DatasetDict, tokenizer: Tokenizer) -> NDArray[np.uint16]:
    tokens: NDArray[np.int64] = np.array(
        [
            tokenizer.eot_token,
            *tokenizer.encode_ordinary(ds_example["text"]),
        ]
    )
    if not ((0 <= tokens).all() and (tokens < 2**16).all()):
        logger.error("Token dictionary too large for uint16. Aborting")
        sys.exit(1)
    t = tokens.astype(np.uint16)
    return t


def get_ds_sample(ds_name: DatasetName, sample: Optional[str]) -> str:
    if not sample:
        s = {
            "fineweb": config.default_fineweb_dataset_sample,
            "smol": config.default_smol_dataset_sample,
        }[ds_name]
        return s
    elif sample not in DS_SAMPLES[ds_name]:
        logger.error(
            f"Invalid dataset sample {sample} for the specified dataset {ds_name}. "
            f"Allowed samples are {", ".join(DS_SAMPLES[ds_name])}. Aborting."
        )
        sys.exit(1)
    return sample


def get_dataset(ds_name: DatasetName, sample: Optional[str]) -> DatasetDict:
    if ds_name == "fineweb":
        return load_dataset(
            config.fineweb_dataset_name,
            name=sample,
            streaming=True,
            split="train",
        )
    elif ds_name == "smol":
        return load_dataset(
            config.smol_dataset_name, streaming=True, name=sample, split="train"
        )


def main() -> None:
    parser = ArgumentParser(
        prog="download_data",
        description="Shard and save the specified dataset to disk",
        usage=f"download_data <ds-name> [DS-SAMPLE]",
    )

    parser.add_argument(
        "ds_name",
        choices=ALLOWED_DATASETS,
        help="The name of the dataset to download.",
    )

    parser.add_argument(
        "--ds-sample",
        dest="ds_sample",
        required=False,
        help="An optional sample of the dataset to download.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--tokenizer-path",
        dest="tokenizer_path",
        required=True,
        help="The tokenizer to tokenize the data with",
        type=str,
    )
    parser.add_argument(
        "--dir-prefix",
        dest="datadir_prefix",
        type=str,
        help="An optional prefix to add to the directory the data will be saved to",
        required=False,
        default=None,
    )

    args = parser.parse_args()
    ds_name: DatasetName = args.ds_name
    dir_prefix = args.datadir_prefix
    _parsed_sample: Optional[str] = args.ds_sample
    sample = get_ds_sample(ds_name, _parsed_sample)

    dirname = f"{dir_prefix}_{sample}" if dir_prefix else sample

    dataset_dir = config.data_dir / ds_name / dirname
    abort_if_dataset_dir_not_empty(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ds = get_dataset(ds_name, sample)

    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        logger.error(
            f"The tokenizer path {tokenizer_path} does not exist.",
            "Please specifiy a valid tokenizer path.",
        )
        sys.exit(1)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    num_procs = max(1, (os.cpu_count() or 1) // 2)
    shard_size = config.dataset_shard_size

    map_fn = partial(tokenize, tokenizer=tokenizer)

    with Pool(num_procs) as p:
        all_tokens = np.zeros((shard_size,), dtype=np.uint16)
        shard = 0
        token_count = 0
        progress_bar: None | tqdm = None

        for tokens in p.imap(map_fn, ds, chunksize=16):
            if (length := len(tokens)) + token_count < shard_size:
                all_tokens[token_count : token_count + length] = tokens
                token_count += length
                progress_bar = update_progress_bar(
                    progress_bar,
                    length,
                    shard_size,
                    shard,
                )
            else:
                remainder = shard_size - token_count
                all_tokens[token_count : token_count + remainder] = tokens[:remainder]
                write_file(dataset_dir, all_tokens, shard)
                update_progress_bar(progress_bar, remainder)
                shard += 1
                progress_bar = None
                all_tokens[0 : length - remainder] = tokens[remainder:]
                token_count = length - remainder

        if token_count != 0:
            write_file(dataset_dir, all_tokens[:token_count], shard)
