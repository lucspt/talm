from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["user", "system", "assistant"]
    content: str
