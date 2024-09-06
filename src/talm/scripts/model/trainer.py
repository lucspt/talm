import time
import inspect
from typing import Any, Generic, Literal, TypeVar, Optional
from logging import Logger
from pathlib import Path
from collections.abc import Iterable

import torch
from gressbar import ProgressBar
from torch.utils.data import DataLoader

from ...model import Model
from ...types import PathLike
from ..helpers import is_file_empty
from ...data.utils import ShardGenerator, ShardedDataLoader
from ...config.model import ModelConfig
from ...lr_scheduler import CosineDecayLR

DataLoaderType = ShardedDataLoader | DataLoader[Any]
DLType = TypeVar("DLType", bound=DataLoaderType)


class BaseTrainer(Generic[DLType]):
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
        train_dataloader: DLType,
        val_dataloader: DLType,
        optimizer: torch.optim.Optimizer,  # type: ignore
        log_file: PathLike,
        checkpoint_dir: PathLike,
        lr_scheduler: CosineDecayLR,
        logger: Logger,
        seed: int,
        gradient_clip_value: float = 1.0,
        logging_strategy: Literal["steps", "epochs"] = "epochs",
        logging_interval: int = 1,
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
            gradient_clip_value (float): The clip value to clip gradient's with.
            logging_strategy (Literal["steps", "epochs"]): Whether to log at the end of epochs or at the end of steps.
            logging_interval (int): Logging, and therefore evaluation, will be performed at this interval.
                This respects `logging_strategy`, so a value of `20` means to log metrics every 20 steps.
        """

        self.device: str
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

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
        self.metrics_to_log = (
            logging_strategy[:-1],
            "train_loss",
            "val_loss",
        )
        self._trainloader_len = len(train_dataloader)
        self.logger = logger
        self.gradient_clip_value = gradient_clip_value
        self.model_config_dict = {
            p: getattr(model.config, p)
            for p in inspect.signature(ModelConfig).parameters
        }
        self.log_interval = logging_interval
        self.log_strategy = logging_strategy

    def _log_msg(self, msg: str, mode: str) -> None:
        with open(self.log_file, mode) as f:
            f.write(" ".join([msg, "\n"]))

    def maybe_init_logging_file(self) -> None:
        if not self.log_file.exists() or is_file_empty(self.log_file):
            self.log_file.touch(exist_ok=True)
            self._log_msg(" ".join(self.metrics_to_log), mode="w")

    def log_metrics(self, **metrics: float | str | int) -> None:
        ms = " ".join([str(metrics.get(m)) for m in self.metrics_to_log])
        self._log_msg(ms, mode="a")

    def create_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "model": self.model.state_dict(),
            "config": self.model_config_dict,
            "optimizer": self.optimizer.state_dict(),
            "step": self.current_step,
            "epoch": epoch,
            "seed": self.seed,
        }
        self.checkpoint_dir.mkdir(exist_ok=True)
        f = self.checkpoint_dir / f"epoch_{epoch}.pt"
        self.logger.info(f"Saving model checkpoint to {f}")
        torch.save(ckpt, f)

    def iter_dataloader(
        self, dataloader: DLType
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError()

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
        for i, (x, y) in enumerate(self.iter_dataloader(self.train_dataloader)):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = model.compute_loss(logits, y)

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
        mean_loss /= i + 1
        return (mean_loss, lr, grad_norm)

    def val_loop(
        self, model: Model, device: str, n_steps: Optional[int] = None
    ) -> float:
        """Perform a validation over `val_dataloader`. Specify

        Returns:
        `float`: The mean loss over all validation of `val_dataloader`.
        """
        model.eval()
        mean_loss = 0.0
        with torch.inference_mode():
            for i, (x, y) in enumerate(self.iter_dataloader(self.val_dataloader)):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = model.compute_loss(logits, y)
                mean_loss += loss.item()
                if i + 1 == n_steps:
                    break
        mean_loss /= i + 1
        return mean_loss

    def on_epoch_start(self, epoch: int) -> None:
        self.progress_bar = ProgressBar(
            self._trainloader_len, bar_prefix=f"Epoch: {epoch}/{self.epochs}"
        )

    def on_train_step_end(self, step: int, train_loss: float) -> None:
        if self.log_strategy == "steps" and step % self.log_interval == 0:
            val_loss = self.val_loop(model=self.model, device=self.device, n_steps=1)
            self.log_metrics(
                step=step,
                train_loss=train_loss,
                val_loss=val_loss,
            )
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

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.create_checkpoint(epoch=epoch)
        if self.log_strategy == "epochs":
            self.log_metrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss)

    def on_train_begin(self) -> None:
        self.maybe_init_logging_file()
        torch.set_float32_matmul_precision("high")

    def train(self) -> None:
        """Train `self.model` over `self.epochs` epochs."""
        self.on_train_begin()
        for epoch in range(self.epochs):
            self.on_epoch_start(epoch=epoch + 1)
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
                epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=lr,
                t=t,
                grad_norm=grad_norm,
            )


class Trainer(BaseTrainer[ShardedDataLoader]):
    """Wrapper class for model pretraining. See `BaseTrainer` for more"""

    def iter_dataloader(self, dataloader: ShardedDataLoader) -> ShardGenerator:
        yield from dataloader.itershards()


class SFTrainer(BaseTrainer[DataLoader[Any]]):
    """Wrapper class for supervised fine tuning. See `BaseTrainer` for more"""

    def iter_dataloader(self, dataloader: DataLoader[Any]) -> DataLoader[Any]:
        return dataloader  # dataloader is already an iterator
