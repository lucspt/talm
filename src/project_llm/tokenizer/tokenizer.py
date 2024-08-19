import regex  # type: ignore

Pair = tuple[int, int]
Decoder = dict[int, bytes]
Merges = dict[Pair, int]


class Tokenizer:
    """A BPE tokenizer for encoding and decoding sequences.

    Attributes:
        decoder (dict[int, bytes]): A mapping specifying integer ids to their corresponding bytes.
        special_tokens (set[str]): A set of special tokens for the tokenizer to consider
    """

    decoder: dict[int, bytes]
    regex_pattern = regex.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )

    def __init__(self, special_tokens: set[str]) -> None:
        self.special_tokens = special_tokens
