import sys
import base64
from typing import Optional
from logging import getLogger
from pathlib import Path

from .tokenizer import Pair, Merges, Decoder, Tokenizer

logger = getLogger(__name__)


class Trainer:
    """A tokenizer trainer class.
    Creates and trains a new tokenizer for encoding and decoding sequences.

    Attributes: See `BaseTokenizer`
    """

    def __init__(self, special_tokens: set[str]) -> None:
        self.regex_pattern = Tokenizer.regex_pattern
        self.special_tokens = special_tokens

    @staticmethod
    def count_pairs(
        tokens: list[int], initial_counts: Optional[dict[Pair, int]] = None
    ) -> dict[Pair, int]:
        """Count the frequency of integer id pairs gen a list of integers.

        Args:
            tokens (list[int]): The integer ids
            initial_counts: (dict[Pair, int], Optional): initial mapping of frequencies to start from.
        """
        counts = {} if initial_counts is None else initial_counts
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

    @staticmethod
    def _build_tokenizer(
        decoder: dict[int, bytes],
        merges: dict[Pair, int],
        special_tokens: Optional[set[str]] = None,
    ) -> tuple[Decoder, Merges]:
        if special_tokens:
            specials_idx = max(decoder) + 1
            specials_to_ids = dict(
                zip(
                    range(specials_idx, specials_idx + len(special_tokens)),
                    (s.encode("utf-8") for s in special_tokens),
                )
            )
            decoder.update(specials_to_ids)
        return decoder, merges

    def train(self, text: str, vocab_size: int, fp: str) -> str:
        """Train a new tokenizer.

        Args:
            text (str): The text to train on.
            vocab_size (int): The desired vocabulary size. Should be an integer greater than 256.
                Traning continues until this number is reached.
            fp (str): A filepath to save the tokenizer to once training is complete.
        """
        if Path(fp).exists():
            logger.error(
                f"Specified filepath '{fp}' already exists, please choose a different filepath."
            )
            sys.exit(1)

        chunks: list[list[int]] = [
            list(c.encode("utf-8")) for c in self.regex_pattern.findall(text)
        ]
        base_size = 256
        decoder = {i: bytes([i]) for i in range(base_size)}
        merges: dict[Pair, int] = {}

        for nth_merge in range(base_size, vocab_size):
            counts: dict[Pair, int] = {}
            for c in chunks:
                self.count_pairs(c, counts)

            pair = max(counts, key=counts.get)  # type: ignore
            chunks = [self.merge(c, pair, nth_merge) for c in chunks]
            merges[pair] = nth_merge
            decoder[nth_merge] = decoder[pair[0]] + decoder[pair[1]]

        decoder, merges = self._build_tokenizer(decoder, merges, self.special_tokens)
        return self.save(fp, decoder, merges)

    @staticmethod
    def decoder_to_textlines(decoder: Decoder) -> list[str]:
        """Textify the decoder mapping for saving"""
        return list(
            f"{base64.b64encode(v).decode("utf-8")} {k}" for k, v in decoder.items()
        )

    def save(self, fp: str, decoder: Decoder, merges: Merges) -> str:
        """Serialize and save the tokenizer to file `fp`"""
        with open(fp, "w") as f:
            f.writelines(
                [
                    "[vocab]\n",
                    *self.decoder_to_textlines(decoder),
                    "\n",
                    "[merges]\n",
                    *[f"{k} {v}" for k, v in merges.items()],
                ]
            )
        return fp
