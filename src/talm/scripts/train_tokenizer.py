from argparse import ArgumentParser

from tokencoder.trainer import TokenizerTrainer

from ..logger import create_logger
from ..config.tokenizer import TokenizerConfig


def main() -> None:
    logger = create_logger(__name__)
    config = TokenizerConfig()

    parser = ArgumentParser(
        prog="train_tokenizer",
        description="Train a tokenizer on a small sample of the FineWeb text dataset",
        usage="train_tokenizer <VOCAB_SIZE> <TOKENIZER_NAME>",
    )
    parser.add_argument(
        "-vs",
        "--vocab-size",
        dest="vocab_size",
        type=int,
        help="The desired vocab size",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="tokenizer_name",
        type=str,
        help="The name of this tokenizer",
        required=True,
    )
    parser.add_argument(
        "--text-file",
        dest="text_file",
        default=config.tokenizer_train_file,
        type=str,
        help="A text file to train the tokenizer on",
    )

    args = parser.parse_args()
    config.registry_dir.mkdir(exist_ok=True)
    vocab_size: int = args.vocab_size
    name: str = args.tokenizer_name
    train_text_file: str = args.text_file

    trainer = TokenizerTrainer(
        name=name,
        special_tokens=config.special_tokens,
    )

    save_dir = config.registry_dir

    with open(train_text_file, "r") as f:
        text = f.read()

    logger.info(f"Training tokenizer with a vocab size of {vocab_size}")
    pth = trainer.train(text=text, vocab_size=vocab_size, save_dir=save_dir)
    logger.info(f"Tokenizer successfully saved to {pth}!")
