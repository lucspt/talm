from typing import Generator
from pathlib import Path
from tempfile import mkdtemp

import torch
import pytest
from tokencoder.trainer import TokenizerTrainer

from talm.resources import Message


@pytest.fixture(scope="module")
def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="class")
def tmp_dir() -> Generator[Path, None, None]:
    "A class-scoped temporary directory"
    d = Path(mkdtemp())
    yield d
    for f in d.iterdir():
        f.unlink()
    d.rmdir()


@pytest.fixture(scope="session")
def session_tmp_dir() -> Generator[Path, None, None]:
    """A session-scoped temporary directory"""
    d = Path(mkdtemp())
    yield d
    for f in d.iterdir():
        f.unlink()
    d.rmdir()


@pytest.fixture(scope="session")
def tokenizer_path(session_tmp_dir: Path) -> str:
    trainer = TokenizerTrainer(name="testing", special_tokens={"<|endoftext|>"})
    path = trainer.train(
        text=Path(__file__).read_text(), vocab_size=270, save_dir=session_tmp_dir
    )
    return path


@pytest.fixture
def messages() -> list[Message]:
    return [
        Message(role="user", content="hey assistant!"),
        Message(role="assistant", content="hey user!"),
    ]
