import sys
import time
from typing import Any
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch import nn
from progress_bar import ProgressBar

from ..data import ShardedDataLoader
from ..model import Model
from ..types import PathLike
from ..config import Config
from ..logger import create_logger
from .helpers import is_file_empty, is_folder_empty
from ..tokenizer import Tokenizer
from ..lr_scheduler import CosineDecayLR

logger = create_logger(__name__)


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


class Trainer:
    """Wrapper class for training a `Model`.

    Attribtues:
        epochs (int): The number of epochs the `train` method will train for.
        model (nn.Module): The model being train.
        train_dataloader (ShardedDataLoader): The dataloader being trained on.
        val_dataloader (ShardedDataLoader): The evaluation dataloader.
        optimizer (torch.optim.Optimizer): The optimizer used when training.
        device (str): The torch device type
        log_file (PathLike): The file to log metrics to.
        checkpoint_dir (PathLike): The directory to save model checkpoints to.
        best_val_loss (float): The best val loss so far achieved during training. Defaults to inf.
    """

    def __init__(
        self,
        epochs: int,
        model: nn.Module,
        model_name: str,
        train_dataloader: ShardedDataLoader,
        val_dataloader: ShardedDataLoader,
        optimizer: torch.optim.Optimizer,  # type: ignore
        log_file: PathLike,
        checkpoint_dir: PathLike,
        lr_scheduler: CosineDecayLR,
    ) -> None:
        """Initialize a model trainer.

        Args:
            epochs (int): The number of epochs to train for.
            model (nn.Module): The model to train.
            train_dataloader (ShardedDataLoader): The dataloader to train on.
            val_dataloader (ShardedDataLoader): The dataloader to perform evaluation on during training.
            optimizer (torch.optim.Optimizer): The optimizer to use when training.
            log_file (PathLike): A string or `Path` pointing to a file to log training metrics to.
            checkpoint_dir (PathLike): A string or `Path` pointing to a file to save model checkpoints to.
        """

        self.device: str
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model_name = model_name
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.log_file = Path(log_file)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.lr_scheduler = lr_scheduler
        self.current_step = 0
        self.best_val_loss = float("inf")
        self.metrics_to_log = ("epoch", "train_loss", "val_loss", "lr")
        self._trainloader_len = len(train_dataloader)

    def _log_msg(self, msg: str, mode: str) -> None:
        with open(self.log_file, mode) as f:
            f.write(" ".join([msg, "\n"]))

    def maybe_init_logging_file(self) -> None:
        if not self.log_file.exists() or is_file_empty(self.log_file):
            self._log_msg(" ".join(self.metrics_to_log), mode="w")

    def log_metrics(self, **metrics: float | str | int) -> None:
        ms = " ".join([str(metrics.get(m)) for m in self.metrics_to_log])
        self._log_msg(ms, mode="a")

    def create_checkpoint(self, filename: str, **kwargs: Any) -> None:
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.current_step,
            **kwargs,
        }
        self.checkpoint_dir.mkdir(exist_ok=True)
        f = self.checkpoint_dir / f"{filename}.pt"
        logger.info(f"Saving model checkpoint to {f}")
        torch.save(ckpt, f)

    def train_loop(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,  # type: ignore
        lr_scheduler: CosineDecayLR,
        device: str,
    ) -> tuple[float, float]:
        """Loop over the `train_dataloader` and train the model.

        Returns:
            ...
        """
        model.train()
        mean_loss = 0.0
        for i, (x, y) in enumerate(self.train_dataloader.itershards()):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lr = lr_scheduler.step()
            lossi = loss.item()
            mean_loss += lossi
            self.current_step += 1
            self.on_train_step_end(step=i + 1, train_loss=lossi, lr=lr)
        mean_loss /= i
        return mean_loss, lr

    def val_loop(
        self,
        model: nn.Module,
        device: str,
    ) -> float:
        """Perform a validation over `val_dataloader`.

            Returns:
        `float`: The mean loss over all validation of `val_dataloader`.
        """
        model.eval()
        mean_loss = 0.0
        with torch.inference_mode():
            for i, (x, y) in enumerate(self.val_dataloader.itershards()):
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                mean_loss += loss.item()
        mean_loss /= i
        return mean_loss

    def on_epoch_start(self, epoch: int) -> None:
        self.progress_bar = ProgressBar(
            self._trainloader_len, bar_prefix=f"Epoch: {epoch}/{self.epochs}"
        )

    def on_train_step_end(self, step: int, train_loss: float, lr: float) -> None:
        self.progress_bar.update(
            step,
            info={"train loss": f"{train_loss:.4f}", "lr": f"{lr:e}"},
            finished=False,
        )

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_loss: float, lr: float, t: float
    ) -> None:
        self.progress_bar.update(
            self._trainloader_len,
            info={
                "train loss": f"{train_loss:.4f}",
                "lr": f"{lr:e}",
                "val loss": f"{val_loss:.4f}",
                "time": f"{t:.4f}",
            },
        )
        self.log_metrics(
            epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss, lr=lr
        )

    def train(self) -> None:
        """Train `self.model` over `self.epochs` epochs."""
        self.maybe_init_logging_file()
        for epoch in range(self.epochs):
            self.on_epoch_start(epoch=epoch + 1)
            t_start = time.time()
            train_loss, lr = self.train_loop(
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
                lr_scheduler=self.lr_scheduler,
            )
            val_loss = self.val_loop(model=self.model, device=self.device)
            t = time.time() - t_start
            self.on_epoch_end(
                epoch + 1, train_loss=train_loss, val_loss=val_loss, lr=lr, t=t
            )


def main() -> None:
    try:
        paths: list[Path] = []
        config = Config()
        parser = ArgumentParser(
            "train_model",
            description="Pre train a language model.",
            usage="train_model <MODEL_NAME> <DATA_DIR>",
        )
        parser.add_argument(
            "-n", "--model-name", dest="model_name", required=True, type=str
        )
        parser.add_argument(
            "-d", "--data-dir", dest="data_dir", required=True, type=str
        )
        args = parser.parse_args()
        model_name: str = args.model_name
        data_dir = args.data_dir

        tokenizer = Tokenizer.from_file(config.tokenizer_path)

        train_dataloader = ShardedDataLoader(
            split="train",
            batch_size=config.batch_size,
            context_len=config.context_len,
            dirname=data_dir,
        )

        val_dataloader = ShardedDataLoader(
            split="val",
            batch_size=config.batch_size,
            context_len=config.context_len,
            dirname=data_dir,
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
                    f"number of model parameters: {n_params:,}" "\n",
                ]
            )
        )

        trainer.train()

    except KeyboardInterrupt:
        catch_err(paths)
    except:
        catch_err(paths)
