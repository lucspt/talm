"""Implements a cosine decayed learning rate scheduler class"""

import math

import torch


class CosineDecayLR:
    """Cosine decayed learning rate scheduler.

    See: https://arxiv.org/pdf/1608.03983v5

    **Note**: This class implements a stepwise scheduler, meaning the learning rate
    is modified at each training step, not each epoch.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,  # type: ignore
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.last_step = 0

    def get_lr(self, step: int) -> float:
        """Get the lr given `step`"""
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        if step > self.max_steps:
            return self.min_lr

        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 * math.cos(decay_ratio * math.pi))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self) -> float:
        """Apply the lr on `self.optimizer`"""
        self.last_step += 1
        lr = self.get_lr(self.last_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
