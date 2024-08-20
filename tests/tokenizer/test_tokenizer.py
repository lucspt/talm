from typing import Generator
from pathlib import Path
from tempfile import mkdtemp

import pytest

from project_llm.tokenizer.config import TokenizerConfig
from project_llm.tokenizer.trainer import Trainer
from project_llm.tokenizer.tokenizer import Tokenizer

config = TokenizerConfig()
special_tokens = {"<|endoftext|>"}

TRAIN_DATA = [
    "hello world",
    "123",
    "testing",
]


@pytest.fixture(scope="module")
def tokenizer_path() -> Generator[Path, None, None]:
    p = Path(mkdtemp()) / "tokenizer.bpe"
    p.parent.mkdir(exist_ok=True)
    yield p
    p.unlink()
    p.parent.rmdir()


@pytest.fixture(scope="module", autouse=True)
def train_tokenizer(tokenizer_path: Path) -> None:
    """Create / train tokenizer and return the file pointing to it"""
    trainer = Trainer()
    trainer.train(text="".join(TRAIN_DATA), vocab_size=300, fp=str(tokenizer_path))


@pytest.fixture(scope="module")
def tokenizer(tokenizer_path: Path, train_tokenizer: None) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_path))


class TestTokenizer:
    def test_count_pairs(self) -> None:
        pairs = [5, 5, 4, 4, 4]
        counts = Tokenizer.count_pairs(pairs)
        assert counts[4, 4] == 2
        assert counts[5, 5] == 1

    def test_count_pairs_with_initial_counts(self) -> None:
        pairs = [4, 4, 4]
        counts = Tokenizer.count_pairs(pairs, {(4, 4): 2})
        assert counts[4, 4] == 4

    def test_merge(self) -> None:
        idx, pair = -1, (1, 1)
        to_merge = [*pair, 2, *pair, 2]
        merged = Tokenizer.merge(to_merge, pair=pair, idx=idx)
        assert merged == [idx, 2, idx, 2]

    def test_encode(self, tokenizer: Tokenizer) -> None:
        tokens = tokenizer.encode("hello world")
        assert isinstance(tokens, list)
        for t in tokens:
            assert isinstance(t, int)

    def test_encode_allow_specials(self, tokenizer: Tokenizer) -> None:
        tokens = tokenizer.encode("".join(special_tokens), allow_special=True)
        assert tokens == [tokenizer.special_tokens_encoder[x] for x in special_tokens]

    @pytest.mark.parametrize("text", TRAIN_DATA)
    def test_decode(self, text: str, tokenizer: Tokenizer) -> None:
        res = tokenizer.decode(tokenizer.encode(text))
        assert isinstance(res, str)
        assert res == text
