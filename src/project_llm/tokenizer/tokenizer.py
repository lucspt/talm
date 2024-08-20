from ast import literal_eval
from base64 import b64decode
from typing import ClassVar, Iterable, Optional

import regex  # type: ignore
from regex import Pattern

from ..types import PathLike

Pair = tuple[int, int]
Decoder = dict[int, bytes]
Merges = dict[Pair, int]


class Tokenizer:
    """A BPE tokenizer for encoding and decoding sequences.

    Attributes:
        decoder (dict[int, bytes]): A mapping specifying integer ids to their corresponding bytes.
        special_tokens (set[str]): A set of special tokens for the tokenizer to consider
        regex
    """

    decoder: dict[int, bytes]
    merges: Merges
    special_tokens_encoder: dict[str, int]
    special_tokens_decoder: dict[int, bytes]
    special_tokens_pattern: str

    regex_pattern: ClassVar[Pattern[str]] = regex.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
    n_vocab: int

    def __init__(
        self,
        decoder: Decoder,
        merges: Merges,
        special_tokens: Optional[Iterable[str]] = None,
    ) -> None:
        self.merges = merges
        self.decoder = decoder
        self.n_vocab = len(decoder) + len(merges)
        self.special_tokens_pattern: Optional[str]
        if special_tokens is not None:
            special_tokens = set(special_tokens).union({"<|endoftext|>"})
        else:
            special_tokens = {"<|endoftext|>"}
        self.__configure_special_tokens(special_tokens)

    def __configure_special_tokens(self, special_tokens: set[str]) -> None:
        dec, enc, n_special = {}, {}, len(special_tokens)
        for nth, spt in zip(
            range(self.n_vocab, self.n_vocab + n_special), special_tokens
        ):
            dec[nth] = spt.encode("utf-8")
            enc[spt] = nth
        self.special_tokens_decoder = dec
        self.special_tokens_encoder = enc
        self.special_tokens_pattern = (
            "(" + "|".join(regex.escape(s) for s in special_tokens) + ")"
        )
        self.n_vocab += n_special

    @staticmethod
    def from_file(
        fp: PathLike, special_tokens: Optional[Iterable[str]] = None
    ) -> "Tokenizer":
        with open(fp, "r") as f:
            content = f.read()
        after_vocab = content.partition("[vocab]\n")[2]
        vocab, _, merges_str = after_vocab.partition("[merges]\n")
        decoder: Decoder = {
            int(v): b64decode(k)
            for k, v in (line.split() for line in vocab.splitlines() if line)
        }
        merges: Merges = {
            literal_eval("".join(pair)): int(token)
            for *pair, token in (
                line.split() for line in merges_str.splitlines() if line
            )
        }
        return Tokenizer(decoder=decoder, merges=merges, special_tokens=special_tokens)

    @staticmethod
    def count_pairs(
        tokens: list[int], counts: Optional[dict[Pair, int]] = None
    ) -> dict[Pair, int]:
        """Count the frequency of integer id pairs gen a list of integers.

        Args:
            tokens (list[int]): The integer ids.
            counts: (dict[Pair, int], Optional): initial mapping of pairs to their counts.
                Further counting performed by this function will mutate the mapping.
        """
        counts = {} if counts is None else counts
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(tokens: list[int], pair: Pair, idx: int) -> list[int]:
        """Replace every occurence of `pair` in `tokens` with `idx`.

        Args:
            tokens (list[int]): The tokens to transform.
            pair (Pair): The pair to replace.
            idx (int): The integer to replace it with
        """
        res, length, i = [], len(tokens), 0
        while i < length:
            if i + 1 < length and (tokens[i], tokens[i + 1]) == pair:
                res.append(idx)
                i += 2
            else:
                res.append(tokens[i])
                i += 1
        return res

    def encode_chunk(self, chunk: bytes) -> list[int]:
        """Tokenize a chunk of bytes

        Args:
            chunk (bytes): The bytes to tokenize
        """
        tokens = list(chunk)
        while len(tokens) >= 2:
            counts = self.count_pairs(tokens)
            pair = min(counts, key=lambda p: counts.get(p, float("inf")))
            if pair in self.merges:
                tokens = self.merge(tokens, pair, self.merges[pair])
            else:
                break
        return tokens

    def encode_ordinary(self, text: str) -> list[int]:
        """Tokenizes bytes without consideration of any special tokens."""
        chunks: list[str] = self.regex_pattern.findall(text)
        tokens: list[int] = []
        for c in chunks:
            tokens.extend(self.encode_chunk(c.encode("utf-8")))
        return tokens

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Tokenize the given text, optionally allowing special tokens to be
        considered.

        Args:
            text (str): The text to tokenize.
            allow_special (bool): Whether or not to consider any of the tokenizer's
                special_tokens when encoding. Defaults to `False`.
        """
        if allow_special and self.special_tokens_pattern:
            tokens = []
            specials_split: list[str] = regex.split(self.special_tokens_pattern, text)
            special_toks = self.special_tokens_encoder
            for tok in specials_split:
                if tok in special_toks:
                    tokens.append(special_toks[tok])
                else:
                    tokens.extend(self.encode_ordinary(tok))
            return tokens
        else:
            return self.encode_ordinary(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode the given tokens to a string."""
        dec, special_dec = self.decoder, self.special_tokens_decoder
        if special_dec:
            bts = [special_dec[t] if t in special_dec else dec[t] for t in tokens]
        else:
            bts = [dec[t] for t in tokens]
        return b"".join(bts).decode("utf-8", errors="replace")
