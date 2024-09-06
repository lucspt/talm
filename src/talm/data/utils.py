from typing import Any, Literal, Optional, Generator
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset  # type: ignore
from torch.utils.data import Dataset

from ..types import PathLike
from ..resources import Message
from ..tokenizer import ChatTokenizer

Split = Literal["train", "val"]
ShardGenerator = Generator[tuple[torch.Tensor, torch.Tensor], None, None]


class TokensShard:
    """Class for loading a dataset shard file and its tokens"""

    def __init__(
        self,
        path: str | Path,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        self.path = path
        self._len: Optional[int] = None
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self) -> int:
        if self._len is None:
            self._len = len(self.tokens())
        return self._len

    def tokens(self) -> torch.Tensor:
        """Load the tokens from the shard"""
        _tokens = torch.tensor(np.load(self.path).astype(np.int32), dtype=torch.long)
        return _tokens

    def _shuffle_example(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shuffle a data example and it's targets."""
        perm = torch.randperm(n=x.size(0), generator=self.generator)
        return x[perm], y[perm]

    def _itershard(self, batch_size: int, context_len: int) -> ShardGenerator:
        """Iterate over the shard's tokens in chunks of `batch_size` where each batch contains `context_len` tokens.
        Returns a tuple of the tokens and their corresponding labels.

        **Note**: each tuple will contain a tensor of shape `(batch_size, context_len)`, if the
        amount of the shard's tokens can not be evenly broken up by this shape, the last batch
        will be dropped.

        Args:
            batch_size (int): How many batches to split the tokens into.
            context_len (int): The amount of tokens to create one example with.
        """
        step, do_shuffle = batch_size * context_len, self.shuffle
        tokens = self.tokens()
        for b in range(0, len(tokens), step):
            buf = tokens[b : b + step + 1]
            if buf.size(0) == step + 1:
                x, y = (
                    buf[:-1].view(batch_size, context_len),
                    buf[1:].view(batch_size, context_len),
                )
                if do_shuffle:
                    yield self._shuffle_example(x, y)
                else:
                    yield x, y


class ShardedDataLoader:
    """Provides utilities for iterating over a directory of text dataset shards."""

    def __init__(
        self,
        split: Split,
        batch_size: int,
        context_len: int,
        dirname: PathLike,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        """Initialize a sharded text dataloader.

        Args:
            split (Literal["train", "val"]): The split of the dataloader.
            batch_size (int): The batch_size used when iterating tokens.
            context_len (int): The number of tokens in a single example of a batch.
            dirname (PathLike): A `Path` or string pointing to the directory of the sharded dataset.
            shuffle (bool): Whether or not to perform shuffling of the dataset. Defaults to `True`.
            generator (torch.Generator, Optional): An optional torch.Generator object to use when generating
                shuffles of the data. Only has an effect when `shuffle` is `True`.
        """
        shardpaths = Path(dirname).glob(f"{split}*")
        self.shards = [
            TokensShard(p, shuffle=shuffle, generator=generator) for p in shardpaths
        ]
        self._total_tokens: Optional[int] = None
        self.batch_size = batch_size
        self.context_len = context_len

    @property
    def total_tokens(self) -> int:
        if self._total_tokens is None:
            self._total_tokens = sum(len(s) for s in self.shards)
        return self._total_tokens

    def __len__(self) -> int:
        tokens_per_batch = self.batch_size * self.context_len
        return self.total_tokens // tokens_per_batch

    def itershards(self) -> ShardGenerator:
        """Iterates over the text shards and yields tensors of shape `(batch_size, context_len)`.

        This generator yields tuples of tensors, each tuple contains tensors of the text tokens and
        their corresponding labels.

        Example:

            ```python
            for x, y in dataloader.itershards():
                print(x.shape, y.shape)
            ```
        """
        b, t = self.batch_size, self.context_len
        for s in self.shards:
            yield from s._itershard(b, t)


# not really sure what generic type PyTorch means to put here, hence the Any
class SFTDataset(Dataset[Any]):
    """A PyTorch dataset for supervised fine tuning"""

    tokens_column_name = "tokens"

    def __init__(
        self, hf_dataset: HFDataset, context_len: int, tokenizer: ChatTokenizer
    ) -> None:
        """Initialize a `SFTDataset`.

        Args:
            hf_dataset (Dataset): A huggingface `Dataset` object to extract tokens from.
            context_len (int): The desired context length.
            tokenizer (ChatTokenizer): The tokenizer to use for encoding the dataset messages.
        """
        self.context_len = context_len
        self.chat_tokens = self.extract_tokens(hf_dataset, tokenizer=tokenizer)

    def dataset_map_fn(
        self,
        ds_example: dict[str, list[Message]],
        tokenizer: ChatTokenizer,
    ) -> dict[str, torch.Tensor]:
        """Process the huggingface dataset"""
        chat_encoded = tokenizer.encode_chat(
            ds_example["messages"], add_generation_prompt=False
        )

        tokens = torch.tensor(chat_encoded, dtype=torch.long)[: self.context_len + 1]
        print(tokens)
        return {self.tokens_column_name: tokens}

    def extract_tokens(
        self, hf_dataset: HFDataset, tokenizer: ChatTokenizer
    ) -> list[list[int]]:
        remove_columns = tuple(c for c in hf_dataset.column_names if c != "messages")
        processed_ds = hf_dataset.map(
            self.dataset_map_fn,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=remove_columns,
        )
        tokens: list[list[int]] = processed_ds[self.tokens_column_name]
        return tokens

    def __len__(self) -> int:
        return len(self.chat_tokens)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(self.chat_tokens[index])
        return tokens[:-1], tokens[1:]
