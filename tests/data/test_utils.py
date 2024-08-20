from typing import Generator
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
import pytest

from project_llm.data.utils import TokensShard, ShardedDataLoader
from project_llm.config.data import DataConfig

config = DataConfig()


@pytest.fixture(scope="module", autouse=True)
def token_shard_path() -> Generator[Path, None, None]:
    p = Path(mkdtemp()) / "train_00001.npy"  # mock shard filename
    p.parent.mkdir(exist_ok=True)
    np.save(p, np.random.randn(config.dataset_shard_size).astype(np.uint16))
    yield p
    p.unlink()
    p.parent.rmdir()


@pytest.mark.slow
class TestTokensShard:
    def test_tokens(self, token_shard_path: Path) -> None:
        shard = TokensShard(token_shard_path)
        assert isinstance(shard.tokens(), torch.Tensor)


@pytest.mark.slow
class TestShardedDataLoader:
    BATCH_SIZE = 32
    CONTEXT_LEN = 8

    @pytest.fixture(scope="class")
    @classmethod
    def dataloader(cls, token_shard_path: Path) -> ShardedDataLoader:
        return ShardedDataLoader(
            split="train",
            batch_size=cls.BATCH_SIZE,
            context_len=cls.CONTEXT_LEN,
            dirname=token_shard_path.parent,
        )

    def test_len(self, dataloader: ShardedDataLoader) -> None:
        assert len(dataloader) == config.dataset_shard_size // (
            self.BATCH_SIZE * self.CONTEXT_LEN
        )

    def test_itershards(self, dataloader: ShardedDataLoader) -> None:
        expected_shape = (self.BATCH_SIZE, self.CONTEXT_LEN)
        for x, y in dataloader.itershards():
            assert x.shape == expected_shape
            assert y.shape == expected_shape
