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

    def test_encode_chat(
        self, tokenizer: ChatTokenizer, messages: list[Message]
    ) -> None:
        tokens = tokenizer.encode_chat(messages)
        assert isinstance(tokens, list)
        assert all(isinstance(x, int) for x in tokens)
