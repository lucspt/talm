from typing import Generator
from pathlib import Path
from dataclasses import asdict

import torch
import pytest
from tokencoder import Tokenizer

from talm.llm import LLM
from talm.model import Model
from talm.resources import Message
from talm.config.model import ModelConfig


class TestLLM:
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
