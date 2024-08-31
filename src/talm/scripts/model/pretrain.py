import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from tokencoder import Tokenizer

from ...data import ShardedDataLoader
from ...model import Model
from .trainer import Trainer
from ...config import Config
from ...logger import create_logger
from ..helpers import is_file_empty, is_folder_empty
from ...lr_scheduler import CosineDecayLR


def catch_err(paths: list[Path]) -> None:
    try:
        for p in paths:
            if p.is_file() and is_file_empty(p):
                p.unlink()
            elif p.is_dir() and is_folder_empty(p):
                p.rmdir()
    except:
        ...


def create_optimizer(
    model: Model,
    weight_decay: float,
    lr: float,
    betas: tuple[float, float],
    eps: float,
) -> torch.optim.Optimizer:  # type: ignore
    """Create an AdamW optimizer with dynamically decayed weights."""
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    # mypy doesn't detect AdamW as an export of torch.optim for some reason
    optim = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)  # type: ignore
    return optim


def main() -> None:
    try:
        logger = create_logger(__name__)
        paths: list[Path] = []
        config = Config()
        parser = ArgumentParser(
            "train_model",
            description="Pre train a language model.",
            usage="train_model <MODEL_NAME> <DATA_DIR> [options]",
        )
        parser.add_argument(
            "-n", "--model-name", dest="model_name", required=True, type=str
        )
        parser.add_argument(
            "-d", "--data-dir", dest="data_dir", required=True, type=str
        )
        parser.add_argument(
            "-t",
            "--tokenizer-path",
            dest="tokenizer_path",
            required=False,
            default=config.tokenizer_dir / "base.json",
            type=str,
        )

        parser.add_argument(
            "-s",
            "--seed",
            dest="seed",
            required=False,
            type=int,
            default=None,
        )
        args = parser.parse_args()
        model_name: str = args.model_name
        data_dir = args.data_dir
        seed: int | None = args.seed
        tokenizer_path: Path = Path(args.tokenizer_path)

        if not tokenizer_path.exists():
            logger.error(
                f"The tokenizer path {tokenizer_path} does not exist. "
                "Please specify a valid tokenizer path"
            )
            sys.exit(1)

        tokenizer = Tokenizer.from_file(tokenizer_path)

        generator = torch.Generator()

        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()
            logger.info(
                f"Seed argument was not specified, using randomly generated seed: {seed}"
            )

        train_dataloader = ShardedDataLoader(
            split="train",
            batch_size=config.batch_size,
            context_len=config.context_len,
            dirname=data_dir,
            shuffle=True,
            generator=generator,
        )

        val_dataloader = ShardedDataLoader(
            split="val",
            batch_size=config.batch_size,
            context_len=config.context_len,
            dirname=data_dir,
            shuffle=False,
        )

        model = Model(
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_block=config.n_transformer_blocks,
            vocab_size=tokenizer.n_vocab,
            context_len=config.context_len,
            dropout=config.dropout,
            ln_eps=config.ln_eps,
        )

        optimizer = create_optimizer(
            model=model,
            weight_decay=config.weight_decay,
            lr=config.min_lr,  # start at min lr.
            betas=config.adam_betas,
            eps=config.adam_eps,
        )

        config.model_dir.mkdir(exist_ok=True)
        checkpoint_dir = config.model_dir / model_name

        if checkpoint_dir.exists() and not is_folder_empty(checkpoint_dir):
            logger.error(
                f"Model checkpoints for {model_name} have already been created at "
                f"{checkpoint_dir}. Aborting training to avoid overwriting. "
                "Please specify a different model name"
            )
            sys.exit(1)

        paths.append(checkpoint_dir)

        config.log_dir.mkdir(exist_ok=True)
        log_file = config.log_dir / f"{model_name}.txt"
        if log_file.exists() and not is_file_empty(log_file):
            logger.error(
                f"A log file for model {model_name} has already been created. "
                "Aborting training to avoid overwriting any data. "
                "Please specify a diferent name"
            )
            sys.exit(1)

        paths.append(log_file)

        lr_max_steps = len(train_dataloader)
        lr_warmup_steps = int(1e7) // (
            config.batch_size * config.context_len
        )  # warmup first 10m tokens

        lr_scheduler = CosineDecayLR(
            optimizer=optimizer,
            warmup_steps=lr_warmup_steps,
            max_steps=lr_max_steps,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
        )

        trainer = Trainer(
            epochs=config.n_epochs,
            log_file=log_file,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            lr_scheduler=lr_scheduler,
            logger=logger,
            seed=seed,
        )

        n_params = sum(p.numel() for p in model.parameters())

        sys.stdout.write(
            "\n".join(
                [
                    "",
                    f"Run Name: {model_name}",
                    "-" * 50,
                    f"model checkpoint directory: {checkpoint_dir}",
                    f"log file location: {log_file}",
                    f"context length: {config.context_len}",
                    f"embedding size: {config.n_embd}",
                    f"batch size: {config.batch_size}",
                    f"n_epochs: {config.n_epochs}",
                    f"number of model parameters: {n_params:,}",
                    "\n",
                ]
            )
        )

        trainer.train()

    except KeyboardInterrupt:
        catch_err(paths)
    except:
        catch_err(paths)
