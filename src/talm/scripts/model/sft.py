from pathlib import Path
from argparse import ArgumentParser

import torch
from datasets import load_dataset  # type: ignore
from tokencoder import Tokenizer

from ...model import Model
from ...resources import Message
from ...tokenizer import ChatTokenizer
from ...config.training import SFTConfig


def ds_map_fn(
    example: dict[str, list[Message]], tokenizer: ChatTokenizer, max_ctx_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    chat_tokens = tokenizer.encode_chat(example["messages"])
    tokens_tensor = torch.tensor(chat_tokens[: (max_ctx_len + 1)])

    return tokens_tensor[:-1], tokens_tensor[1:]


def main() -> None:
    parser = ArgumentParser(
        "sft",
        usage="sft [OPTIONS]",
        description="Perform supervised fine tuning given a model checkpoint",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        required=True,
        type=Path,
        help="The file location of the model checkpoint",
    )
    parser.add_argument(
        "-t",
        "--tokenizer-path",
        dest="tokenizer_path",
        default="tokenizers/base.json",
        type=str,
        help="The file location of the tokenizer to use",
    )

    args = parser.parse_args()
    model_path: Path = args.model_path
    tokenizer_path: str = args.tokenizer_path

    model_checkpoint = torch.load(model_path, weights_only=True)
    config = model_checkpoint["config"]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = Model(config=model_checkpoint["config"], vocab_size=tokenizer.n_vocab)
    model.load_state_dict(model_checkpoint["model"])

    config = SFTConfig()

    optimizer = torch.optim.AdamW(  # type: ignore
        model.parameters(),
        lr=config.min_lr,
        betas=config.adam_betas,
        weight_decay=config.weight_decay,
        eps=config.adam_eps,
    )

    ds = load_dataset(config.dataset_name, name="default")
