from typing import Literal, Generator
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig

Split = Literal["train", "val"]
ShardGenerator = Generator[tuple[torch.Tensor, torch.Tensor], None, None]
config = DataConfig()


class TokensShard:
    """Class for loading a dataset shard file and its tokens"""

    def __init__(self, path: str | Path):
        self.path = path
        self._len = len(self.tokens())

    def __len__(self) -> int:
        return self._len

    def tokens(self) -> torch.Tensor:
        """Load the tokens from the shard"""
        _tokens = torch.tensor(np.load(self.path).astype(np.int32), dtype=torch.long)
        return _tokens

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
        step = batch_size * context_len
        tokens = self.tokens()
        for b in range(0, len(tokens), step):
            buf = tokens[b : b + step + 1]
            if buf.size(0) == step + 1:
                yield (
                    buf[:-1].view(batch_size, context_len),
                    buf[1:].view(batch_size, context_len),
                )


class ShardedDataLoader:
    """Provides utilities for iterating over a directory of text dataset shards."""

    def __init__(
        self,
        split: Split,
        batch_size: int,
        context_len: int,
        dirname: str | Path = config.save_dir,
    ):
        """Initialize a sharded text dataloader.

        Args:
            split (Literal["train", "val"]): The split of the dataloader.
            batch_size (int): The batch_size used when iterating tokens.
            context_len (int): The number of tokens in a single example of a batch.
        """
        shardpaths = Path(dirname).glob(f"{split}*")
        self.shards = [TokensShard(p) for p in shardpaths]
        self._total_tokens = sum(len(s) for s in self.shards)
        self.batch_size = batch_size
        self.context_len = context_len

    def __len__(self) -> int:
        tokens_per_batch = self.batch_size * self.context_len
        return self._total_tokens // tokens_per_batch

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
