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
from talm.resources import Message
from talm.config.model import ModelConfig
from talm.config.tokenizer import TokenizerConfig


@pytest.fixture(scope="class")
def tmp_dir() -> Generator[Path, None, None]:
    d = Path(mkdtemp())
    yield d
    for f in d.iterdir():
        f.unlink()
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
            n_embd=16, n_head=4, context_len=128, dropout=0.0, n_transformer_layers=2
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

    @pytest.mark.parametrize("temperature,device", [(1.0, "cpu"), (None, None)])
    def test_generate_output(
        self,
        llm: LLM,
        temperature: float | None,
        device: str | None,
    ) -> None:
        out = llm.generate(
            prompt_tokens=list(range(10)),
            temperature=temperature,
            device=device,
        )
        assert isinstance(out, list)
        for x in out:
            assert isinstance(x, int)

    @pytest.mark.parametrize("max_tokens", (1, 4))
    def test_generate_max_tokens(self, llm: LLM, max_tokens: int) -> None:
        out = llm.generate(prompt_tokens=list(range(10)), max_tokens=max_tokens)
        assert len(out) == max_tokens

    @pytest.fixture
    def messages(self) -> list[Message]:
        return [
            Message(role="user", content="hey assistant!"),
            Message(role="assistant", content="hey user!"),
        ]

    def test_encode_chat_message(self, llm: LLM, messages: list[Message]) -> None:
        tokens = llm.encode_chat_message(messages[0])
        assert isinstance(tokens, list)
        for x in tokens:
            assert isinstance(x, int)
        assert llm.tokenizer.eot_token in tokens
        assert "\n" in llm.tokenizer.decode(tokens)

    def test_encode_chat(self, llm: LLM, messages: list[Message]) -> None:
        tokens = llm.encode_chat(messages)
        assert isinstance(tokens, list)
        assert all(isinstance(x, int) for x in tokens)

    def test_chat_completion(self, llm: LLM, messages: list[Message]) -> None:
        completion = llm.chat_completion(messages)
        assert "role" in completion
        assert completion["role"] == "assistant"
        assert "content" in completion
        assert isinstance(completion["content"], str)

    def test_chat_completion_raises_when_ctx_len_too_long(
        self, llm: LLM, messages: list[Message]
    ) -> None:
        long_chat = [
            *messages,
            Message(role="user", content="text that will be long" * 1000),
        ]
        with pytest.raises(Exception):
            llm.chat_completion(long_chat)
