import time
from typing import Generator
from logging import Logger
from pathlib import Path
from contextlib import contextmanager

import torch
from gressbar import ProgressBar

from ...data import ShardedDataLoader
from ...model import Model
from ...types import PathLike
from ..helpers import is_file_empty
from ...lr_scheduler import CosineDecayLR


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
        model: Model,
        model_name: str,
        train_dataloader: ShardedDataLoader,
        val_dataloader: ShardedDataLoader,
        optimizer: torch.optim.Optimizer,  # type: ignore
        log_file: PathLike,
        checkpoint_dir: PathLike,
        lr_scheduler: CosineDecayLR,
        logger: Logger,
        seed: int,
        gradient_clip_value: float = 1.0,
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
            logger (Logger): A python Logger object to perform logging.
        """

        if torch.cuda.is_available():
            self.device_type = "cuda"
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            self.device_type = "mps"
        else:
            self.device_type = "cpu"

        self._is_autocast_available: bool = torch.amp.is_autocast_available(  # type: ignore
            device_type=self.device_type
        )

        self.device = self.device_type
        self.seed = seed
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
        self.logger = logger
        self.gradient_clip_value = gradient_clip_value

    def _log_msg(self, msg: str, mode: str) -> None:
        with open(self.log_file, mode) as f:
            f.write(" ".join([msg, "\n"]))

    def maybe_init_logging_file(self) -> None:
        if not self.log_file.exists() or is_file_empty(self.log_file):
            self._log_msg(" ".join(self.metrics_to_log), mode="w")

    def log_metrics(self, **metrics: float | str | int) -> None:
        ms = " ".join([str(metrics.get(m)) for m in self.metrics_to_log])
        self._log_msg(ms, mode="a")

    def create_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.current_step,
            "epoch": epoch,
            "seed": self.seed,
        }
        self.checkpoint_dir.mkdir(exist_ok=True)
        f = self.checkpoint_dir / f"{epoch:06d}.pt"
        self.logger.info(f"Saving model checkpoint to {f}")
        torch.save(ckpt, f)

    @contextmanager
    def compute_loss_context_manager(self) -> Generator[None, None, None]:
        if self._is_autocast_available:
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                yield None
        else:
            yield None

    def compute_loss(
        self, x: torch.Tensor, y: torch.Tensor, model: Model
    ) -> torch.Tensor:
        """Do a forward pass and compute the loss"""
        with self.compute_loss_context_manager():
            logits = self.model(x)
            loss = model.compute_loss(logits, y)
        return loss

    def train_loop(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,  # type: ignore
        lr_scheduler: CosineDecayLR,
        device: str,
    ) -> tuple[float, float, float]:
        """Loop over the `train_dataloader` and train the model.

        Returns:
            ...
        """
        model.train()
        mean_loss = 0.0
        for i, (x, y) in enumerate(self.train_dataloader.itershards()):
            x, y = x.to(device), y.to(device)
            loss = self.compute_loss(x, y, model=model)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.gradient_clip_value
            )
            optimizer.step()
            lr = lr_scheduler.step()
            lossi = loss.item()
            mean_loss += lossi
            self.current_step += 1
            self.on_train_step_end(step=i + 1, train_loss=lossi)
        mean_loss /= i
        return (mean_loss, lr, grad_norm)

    def val_loop(
        self,
        model: Model,
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
                loss = self.compute_loss(x, y, model=model)
                mean_loss += loss.item()
        mean_loss /= i
        return mean_loss

    def on_epoch_start(self, epoch: int) -> None:
        self.progress_bar = ProgressBar(
            self._trainloader_len, bar_prefix=f"Epoch: {epoch}/{self.epochs}"
        )

    def on_train_step_end(self, step: int, train_loss: float) -> None:
        self.progress_bar.update(
            step,
            info={"train loss": f"{train_loss:.4f}"},
            finished=False,
        )

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        t: float,
        grad_norm: float,
    ) -> None:
        self.progress_bar.update(
            self._trainloader_len,
            info={
                "train loss": f"{train_loss:.4f}",
                "lr": f"{lr:e}",
                "grad norm": f"{grad_norm:.3f}",
                "val loss": f"{val_loss:.4f}",
                "time": f"{t:.1f}s",
            },
        )
        self.log_metrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss, lr=lr)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.create_checkpoint(epoch=epoch)

    def _set_default_dtype(self) -> None:
        if self.device_type == "cuda" and torch.cuda.is_bf16_supported():
            torch.set_default_dtype(torch.bfloat16)  # type: ignore

    def on_train_begin(self) -> None:
        self.maybe_init_logging_file()
        torch.set_float32_matmul_precision("high")
        self._set_default_dtype()

    def train(self) -> None:
        """Train `self.model` over `self.epochs` epochs."""
        self.on_train_begin()
        for epoch in range(1, self.epochs + 1):
            self.on_epoch_start(epoch=epoch)
            t_start = time.time()
            train_loss, lr, grad_norm = self.train_loop(
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
                lr_scheduler=self.lr_scheduler,
            )
            val_loss = self.val_loop(model=self.model, device=self.device)
            t = time.time() - t_start
            self.on_epoch_end(
                epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=lr,
                t=t,
                grad_norm=grad_norm,
            )
