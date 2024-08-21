import sys
import base64
from typing import Optional
from logging import Logger
from argparse import ArgumentParser

from ..types import PathLike
from ..logger import create_logger
from .helpers import is_file_empty
from ..config.tokenizer import TokenizerConfig
from ..tokenizer.tokenizer import Pair, Merges, Decoder, Tokenizer


class Trainer:
    """A tokenizer trainer class.
    Creates and trains a new tokenizer for encoding and decoding sequences.
    """

    def __init__(self) -> None:
        self.regex_pattern = Tokenizer.regex_pattern

    def train_and_save(
        self,
        text: str,
        vocab_size: int,
        fp: PathLike,
        logger: Optional[Logger] = None,
    ) -> str:
        """Train a save a tokenizer to `fp`.

        Args:
            text (str): The text to train on.
            vocab_size (int): The desired vocabulary size. Should be an integer greater than 256.
                Traning continues until this number is reached.
            fp (PathLike): A string or `Path` pointing to a file to save the tokenizer to.
            logger (Logger): A logger object for logging during training.
        """
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
                if logger:
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

        return self.save(fp, decoder=decoder, merges=merges)

    @staticmethod
    def decoder_to_textlines(decoder: Decoder) -> list[str]:
        """Textify the decoder mapping"""
        return list(
            f"{base64.b64encode(v).decode("utf-8")} {k}\n" for k, v in decoder.items()
        )

    def save(self, fp: PathLike, decoder: Decoder, merges: Merges) -> str:
        """Serialize and save the tokenizer to `fp`"""
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
        return str(fp)


def main() -> None:
    logger = create_logger(__name__)
    config = TokenizerConfig()

    parser = ArgumentParser(
        prog="train_tokenizer",
        description="Train a tokenizer on a small sample of the FineWeb text dataset",
        usage="train_tokenizer <VOCAB_SIZE> <TOKENIZER_NAME>",
    )
    parser.add_argument("-vs", "--vocab_size", dest="vocab_size", type=int)
    parser.add_argument("-n", "--name", dest="tokenizer_name", type=str)
    parser.add_argument(
        "--text_file", dest="text_file", default=config.tokenizer_train_file, type=str
    )

    args = parser.parse_args()
    trainer = Trainer()
    config.registry_dir.mkdir(exist_ok=True)
    vocab_size: int = args.vocab_size
    name: str = args.tokenizer_name
    train_text_file: str = args.text_file

    fp = config.registry_dir / f"{name}.bpe"
    if fp.exists() and not is_file_empty(fp):
        logger.error(
            f"A tokenizer with the name {name} has already been  "
            f"trained and saved to the file {fp}. Aborting the training to avoid "
            "overwriting any data. Please specify a different name."
        )
        sys.exit(1)

    with open(train_text_file, "r") as f:
        text = f.read()[: int(1e6)]

    logger.info(f"Training tokenizer with a vocab size of {vocab_size}")
    trainer.train_and_save(text=text, vocab_size=vocab_size, fp=fp, logger=logger)
    logger.info(f"Tokenizer successfully saved to {fp}!")
