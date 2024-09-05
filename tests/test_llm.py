from typing import Generator
from pathlib import Path
from tempfile import mkdtemp
from dataclasses import asdict

import torch
import pytest
from tokencoder import Tokenizer
from tokencoder.trainer import TokenizerTrainer

from talm.llm import LLM
from talm.model import Model
from talm.config.model import ModelConfig
from talm.config.tokenizer import TokenizerConfig


@pytest.fixture(scope="class")
def tmp_dir() -> Generator[Path, None, None]:
    d = Path(mkdtemp())
    yield d
    d.rmdir()


class TestLLM:
    @pytest.fixture(autouse=True, scope="class")
    def tokenizer_path(self, tmp_dir: Path) -> Generator[str, None, None]:
        trainer = TokenizerTrainer(
            name="testing", special_tokens=TokenizerConfig().special_tokens
        )
        path = trainer.train(
            text=Path(__file__).read_text(), vocab_size=270, save_dir=tmp_dir
        )
        yield path
        Path(path).unlink()

    @pytest.fixture(scope="class")
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            n_embd=16, n_head=4, context_len=32, dropout=0.0, n_transformer_layers=2
        )  # very lite model for testing

    @pytest.fixture(autouse=True, scope="class")
    def model_checkpoint(
        self, tokenizer_path: str, tmp_dir: Path, model_config: ModelConfig
    ) -> Generator[Path, None, None]:
        t = Tokenizer.from_file(tokenizer_path)
        model = Model(config=model_config, vocab_size=t.n_vocab)
        f = tmp_dir / "model.pt"
        torch.save({"model": model.state_dict()}, f)
        yield f
        f.unlink()

    @pytest.fixture
    def llm(
        self, model_checkpoint: Path, tokenizer_path: str, model_config: ModelConfig
    ) -> LLM:
        return LLM.build(model_checkpoint, tokenizer_path, **asdict(model_config))

    def test_generate_output(self, llm: LLM) -> None:
        out = llm.generate(prompt_tokens=list(range(10)))
        assert isinstance(out, list)
        for x in out:
            assert isinstance(x, int)

    @pytest.mark.parametrize("max_tokens", (1, 4))
    def test_generate_max_tokens(self, llm: LLM, max_tokens: int) -> None:
        out = llm.generate(prompt_tokens=list(range(10)), max_tokens=max_tokens)
        assert len(out) == max_tokens
