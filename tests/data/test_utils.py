from typing import Generator
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
import pytest
from datasets import Dataset as HFDataset, load_dataset  # type: ignore
from pytest_mock import MockerFixture

from talm.tokenizer import ChatTokenizer
from talm.data.utils import (
    SFTDataset,
    TokensShard,
    ShardedDataLoader,
)
from talm.config.data import DataConfig
from talm.config.training import SFTConfig

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


@pytest.mark.slow
class TestSFTDataset:
    context_len = 12
    ds_name = SFTConfig().dataset_name
    select_len = context_len * 2

    @pytest.fixture(scope="class")
    def hf_ds(self) -> HFDataset:
        return load_dataset(self.ds_name, name="default", split="train_sft").select(
            range(0, self.select_len)
        )

    @pytest.fixture(scope="class")
    def ds(self, tokenizer_path: str, hf_ds: HFDataset) -> SFTDataset:
        tokenizer = ChatTokenizer.from_file(tokenizer_path)
        ds = SFTDataset(hf_ds, self.context_len, tokenizer=tokenizer)
        return ds

    def test_iterable(self, ds: SFTDataset) -> None:
        for x, y in ds:  # type: ignore
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            assert x.dtype == torch.long
            assert y.dtype == torch.long
            assert torch.eq(x[1:], y[:-1]).all().item()

    def test_len(self, ds: SFTDataset) -> None:
        assert len(ds) == self.select_len

    def test_dataloader_collate_fn(self) -> None:
        max_len = 32
        rand_max_len = torch.randn((max_len,))
        rand_to_truncate = torch.randn((max_len * 2))
        example = [(rand_max_len, rand_max_len), (rand_to_truncate, rand_to_truncate)]
        collated = SFTDataset.dataloader_collate_fn(example)
        for x, y in collated:
            assert x.size(0) == max_len and y.size(0) == max_len

    def test_get_dataloader(self, ds: SFTDataset, mocker: MockerFixture) -> None:
        import talm.data.utils as dl_spy

        spy = mocker.spy(dl_spy, "DataLoader")
        gen = torch.Generator()
        ds.get_dataloader(batch_size=16, shuffle=True, generator=gen)
        spy.assert_called_once_with(
            ds,
            batch_size=16,
            shuffle=True,
            collate_fn=ds.dataloader_collate_fn,
            generator=gen,
        )

    def test_dataloader_can_iterate_over_baches(self, ds: SFTDataset) -> None:
        loader = ds.get_dataloader(2, True)
        for _ in loader:
            ...
