import sys
from logging import Logger
from pathlib import Path

import torch

from ...model import Model
from ...types import PathLike
from ...config import Config
from ..helpers import is_file_empty, is_folder_empty
from ...config.training import TrainingConfig


def catch_interruption(paths: list[Path]) -> None:
    """Delete any empty files on Exception or KeyboardInterrupt"""
    try:
        for p in paths:
            if p.exists():
                if p.is_file() and is_file_empty(p):
                    p.unlink()
                elif p.is_dir() and is_folder_empty(p):
                    p.rmdir()
    except:
        ...


def create_optimizer(model: Model, config: TrainingConfig) -> torch.optim.AdamW:  # type: ignore
    """Create an AdamW optimizer with dynamically decayed weights."""
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    # mypy doesn't detect AdamW as an export of torch.optim for some reason
    optim = torch.optim.AdamW(  # type: ignore
        param_groups, lr=config.min_lr, betas=config.adam_betas, eps=config.adam_eps
    )
    return optim


def assert_model_checkpoint_does_not_exist(
    checkpoint_dir: Path, logger: Logger
) -> None:
    """Make sure checkpoint directory is non existant"""
    if checkpoint_dir.exists() and not is_folder_empty(checkpoint_dir):
        logger.error(
            f"Model checkpoint directory for `{checkpoint_dir.name}` has already been created at "
            f"{checkpoint_dir}. Aborting training to avoid overwriting. "
            "Please specify a different model name"
        )
        sys.exit(1)


def get_seeded_generator(
    seed: int | None, logger: Logger
) -> tuple[torch.Generator, int]:
    """Create a generator and seed it"""
    generator = torch.Generator()

    if seed is not None:
        generator.manual_seed(seed)
    else:
        seed = generator.seed()
        logger.info(
            f"Seed argument was not specified, using randomly generated seed: {seed}"
        )
    return generator, seed


def count_params(model: torch.nn.Module) -> int:
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters())


def print_training_run(
    run_name: str,
    n_params: int,
    checkpoint_dir: PathLike,
    log_file: PathLike,
    config: Config,
) -> None:
    """Print a training run summary"""
    sys.stdout.write(
        "\n".join(
            [
                "",
                f"Run Name: {run_name}",
                "-" * 50,
                f"model checkpoint directory: {checkpoint_dir}",
                f"log file location: {log_file}",
                f"context length: {config.context_len}",
                f"embedding size: {config.n_embd}",
                f"batch size: {config.batch_size}",
                f"number of epochs: {config.n_epochs}",
                f"number of model parameters: {n_params:,}",
                "\n",
            ]
        )
    )
