from pathlib import Path
from argparse import ArgumentParser

from ..config.tokenizer import TokenizerConfig
from ..tokenizer.trainer import Trainer

config = TokenizerConfig()

parser = ArgumentParser(
    prog="Tokenizer trainer",
    description="Train a tokenizer on a small sample of the FineWeb text dataset",
)
parser.add_argument(
    "-vs", "--vocab_size", dest="vocab_size", default=config.vocab_size, type=int
)
parser.add_argument(
    "-f", "--filename", dest="filename", default=config.save_path, type=str
)
args = parser.parse_args()


def main() -> None:
    trainer = Trainer()
    config.save_path.parent.mkdir(exist_ok=True)
    vocab_size, filename = args.vocab_size, Path.cwd() / args.filename
    with open(config.train_text_file, "r") as f:
        text = f.read()[: int(1e6)]
    trainer.train(text=text, vocab_size=vocab_size, fp=filename)
