import pytest
from tokencoder import Tokenizer

from talm.resources import Message
from talm.tokenizer import ChatTokenizer


class TestChatTokenizer:
    @pytest.fixture(scope="class")
    def tokenizer(self, tokenizer_path: str) -> ChatTokenizer:
        return ChatTokenizer(Tokenizer.from_file(tokenizer_path))

    def test_encode_chat_message(
        self, tokenizer: ChatTokenizer, messages: list[Message]
    ) -> None:
        tokens = tokenizer.encode_chat_message(messages[0])
        assert isinstance(tokens, list)
        for x in tokens:
            assert isinstance(x, int)
        assert tokenizer.tokenizer.eot_token in tokens
        assert "\n" in tokenizer.tokenizer.decode(tokens)

    @pytest.mark.parametrize("add_generation_prompt", (False, True))
    def test_encode_chat(
        self,
        tokenizer: ChatTokenizer,
        messages: list[Message],
        add_generation_prompt: bool,
    ) -> None:
        tokens = tokenizer.encode_chat(
            messages, add_generation_prompt=add_generation_prompt
        )
        assert isinstance(tokens, list)
        assert all(isinstance(x, int) for x in tokens)
        generation_prompt = "assistant\n"
        has_generation_promp = (
            tokenizer.tokenizer.decode(tokens)[-len(generation_prompt) :]
            == generation_prompt
        )
        assert has_generation_promp == add_generation_prompt
