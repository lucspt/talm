import sys
import base64
from logging import INFO, getLogger, basicConfig
from pathlib import Path

from .config import TokenizerConfig
from .tokenizer import Pair, Merges, Decoder, Tokenizer

config = TokenizerConfig()

basicConfig()
logger = getLogger(__name__)
logger.setLevel(INFO)


class Trainer:
    """A tokenizer trainer class.
    Creates and trains a new tokenizer for encoding and decoding sequences.
    """

    def __init__(self) -> None:
        self.regex_pattern = Tokenizer.regex_pattern

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

        logger.info(f"Training tokenizer with a vocab size of {vocab_size}")
        chunks: list[list[int]] = [
            list(c.encode("utf-8")) for c in self.regex_pattern.findall(text)
        ]
        base_size = 256
        decoder = {i: bytes([i]) for i in range(base_size)}
        merges: dict[Pair, int] = {}

        for nth_merge in range(base_size, vocab_size):
            counts: dict[Pair, int] = {}
            for c in chunks:
                Tokenizer.count_pairs(c, counts)

            if not counts:  # we've already done all possible merges
                max_size = len(decoder)
                logger.warning(
                    "NOTE: The given text for tokenizer training"
                    f"is too short for the specified vocab_size, {vocab_size}. "
                    f"The max vocab size for the text is {max_size}, "
                    f"and thus the tokenizer's vocab size will be {max_size}"
                )
                break

            pair = max(counts, key=counts.get)  # type: ignore
            chunks = [Tokenizer.merge(c, pair, nth_merge) for c in chunks]
            merges[pair] = nth_merge
            decoder[nth_merge] = decoder[pair[0]] + decoder[pair[1]]

        return self.save(fp, decoder, merges)

    @staticmethod
    def decoder_to_textlines(decoder: Decoder) -> list[str]:
        """Textify the decoder mapping"""
        return list(
            f"{base64.b64encode(v).decode("utf-8")} {k}\n" for k, v in decoder.items()
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
                    *[f"{k} {v}\n" for k, v in merges.items()],
                ]
            )

        logger.info(f"Tokenizer data saved to {fp}")
        return fp
