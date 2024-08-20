import os
import sys
from typing import Optional
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm  # type: ignore
from datasets import DatasetDict, load_dataset  # type: ignore
from numpy.typing import NDArray, ArrayLike

from ..logger import create_logger
from ..config.data import DataConfig
from ..config.root import RootConfig
from ..tokenizer.tokenizer import Tokenizer

config = DataConfig()

logger = create_logger(__name__)


def abort_if_data_dir_not_empty() -> None:
    save_dir = config.dataset_dir
    if save_dir.exists() and len(list(save_dir.glob("*"))) > 0:
        logger.error(
            f"Dataset dir '{save_dir}' already exists, and is not empty. "
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
        bar = tqdm(total=total, desc=f"Shard {nth_shard}")
    bar.update(n)
    return bar


def write_file(dir: Path, tokens: ArrayLike, shard: int) -> None:
    split_prefix = "val" if shard == 0 else "train"
    shard = 1 if split_prefix == "val" else shard
    pth = dir / f"{split_prefix}_{shard:06d}"
    logger.info(f"Saving new shard at {pth}")
    np.save(pth, tokens)


tokenizer = Tokenizer.from_file(RootConfig.tokenizer_path)


def tokenize(ds_example: DatasetDict) -> NDArray[np.uint16]:
    tokens: NDArray[np.int64] = np.array(
        [
            tokenizer.special_tokens_encoder["<|endoftext|>"],
            *tokenizer.encode_ordinary(ds_example["text"]),
        ]
    )
    if not ((0 <= tokens).all() and (tokens < 2**16).all()):
        logger.error("Token dictionary too large for uint16. Aborting")
        sys.exit(1)
    t = tokens.astype(np.uint16)
    return t


def main() -> None:
    abort_if_data_dir_not_empty()
    config.dataset_dir.mkdir(exist_ok=True, parents=True)
    ds = load_dataset(
        config.dataset_name, name=config.dataset_sample, streaming=True, split="train"
    )
    num_procs = max(1, (os.cpu_count() or 1) // 2)
    shard_size = config.dataset_shard_size
    with Pool(num_procs) as p:
        all_tokens = np.zeros((shard_size,), dtype=np.uint16)
        shard = 0
        token_count = 0
        progress_bar: None | tqdm = None

        for tokens in p.imap(tokenize, ds, chunksize=16):
            if (length := len(tokens)) + token_count < shard_size:
                all_tokens[token_count : token_count + length] = tokens
                token_count += length
                progress_bar = update_progress_bar(
                    progress_bar, length, shard_size, shard
                )
            else:
                remainder = shard_size - token_count
                all_tokens[token_count : token_count + remainder] = tokens[:remainder]
                write_file(config.dataset_dir, all_tokens, shard)
                update_progress_bar(progress_bar, remainder)
                shard += 1
                progress_bar = None
                all_tokens[0 : length - remainder] = tokens[remainder:]
                token_count = length - remainder

        if token_count != 0:
            write_file(config.dataset_dir, all_tokens[:token_count], shard)
