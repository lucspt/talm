from typing import Generator
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
import pytest

from talm.data.utils import TokensShard, ShardedDataLoader
from talm.config.data import DataConfig

config = DataConfig()


SHARD_SIZE = int(1e6)


@pytest.fixture(scope="module", autouse=True)
def token_shard_path() -> Generator[Path, None, None]:
    p = Path(mkdtemp()) / "train_00001.npy"  # mock shard filename
    p.parent.mkdir(exist_ok=True)
    np.save(p, np.random.randint(low=0, high=1000, size=(SHARD_SIZE,), dtype=np.uint16))
    yield p
    p.unlink()
    p.parent.rmdir()


@pytest.mark.slow
class TestTokensShard:
    @pytest.fixture
    def shard(self, token_shard_path: Path) -> TokensShard:
        return TokensShard(token_shard_path)

    def test_tokens(self, shard: TokensShard) -> None:
        assert isinstance(shard.tokens(), torch.Tensor)

    def test_shuffle_examples(self, token_shard_path: Path, shard: TokensShard) -> None:
        b, t = 32, 8
        batchx, batchy = torch.randn((b, t)), torch.randn((b, t))
        mapping = dict((tuple(x.tolist()), y) for x, y in zip(batchx, batchy))
        shuffledx, shuffledy = shard._shuffle_example(batchx, batchy)
        for x, y in zip(shuffledx, shuffledy):
            assert (mapping[tuple(x.tolist())] == y).all().item() == True


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
        assert len(dataloader) == SHARD_SIZE // (self.BATCH_SIZE * self.CONTEXT_LEN)

    def test_itershards(self, dataloader: ShardedDataLoader) -> None:
        expected_shape = (self.BATCH_SIZE, self.CONTEXT_LEN)
        for x, y in dataloader.itershards():
            assert x.shape == expected_shape
            assert y.shape == expected_shape

    def test_itershards_shuffled(
        self, token_shard_path: Path, dataloader: ShardedDataLoader
    ) -> None:
        dataloader_noshuffle = ShardedDataLoader(
            split="train",
            batch_size=self.BATCH_SIZE,
            context_len=self.CONTEXT_LEN,
            dirname=token_shard_path.parent,
            shuffle=False,
        )

        zipped_shards = zip(dataloader.itershards(), dataloader_noshuffle.itershards())
        assert any(
            (x_shuff != x).all().item() for (x, _), (x_shuff, _) in zipped_shards
        )
