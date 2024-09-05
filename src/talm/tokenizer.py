from tokencoder import Tokenizer

from .resources import Message


class ChatTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode_chat_message(self, message: Message) -> list[int]:
        """Encode a `Message` dict to tokens."""
        content = "\n".join([message["role"], message["content"].strip()])
        return [
            *self.tokenizer.encode_ordinary(content),
            self.tokenizer.eot_token,
            *self.tokenizer.encode("\n"),
        ]

    def encode_chat(
        self, chat: list[Message], add_generation_prompt: bool = True
    ) -> list[int]:
        """Encode a list of `Message` dicts to tokens."""
        tokens = []
        for msg in chat:
            tokens.extend(self.encode_chat_message(msg))
        if add_generation_prompt:
            tokens.extend(self.tokenizer.encode_ordinary("assistant\n"))
        return tokens
