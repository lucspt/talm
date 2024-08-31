import torch
import pytest
from torch import nn

from talm.lr_scheduler import CosineDecayLR


class TestCosineDecayLR:
    max_lr = 1e-4
    min_lr = 3e-4
    warmup_steps = 275
    max_steps = 400

    @staticmethod
    def allclose(x: float, y: float) -> bool:
        print(x, y)
        return torch.allclose(torch.tensor(x), torch.tensor(y))

    @pytest.fixture()
    def scheduler(self) -> CosineDecayLR:
        return CosineDecayLR(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            optimizer=torch.optim.AdamW(params=nn.Linear(32, 32).parameters()),  # type: ignore
        )

    def test_get_lr_after_max_steps(self, scheduler: CosineDecayLR) -> None:
        lrs = [scheduler.get_lr(s) for s in range(self.max_steps + 100)][
            self.max_steps + 1 :
        ]
        for x in lrs:
            assert self.allclose(x, self.min_lr)

    def test_step(self, scheduler: CosineDecayLR) -> None:
        optimizer = torch.optim.AdamW(params=nn.Linear(32, 32).parameters(), lr=1.0)  # type: ignore
        scheduler = CosineDecayLR(
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            optimizer=optimizer,
        )
        for step in range(1, 5):
            scheduler.step()
            expected_lr = scheduler.get_lr(step)
            for p in optimizer.param_groups:
                assert self.allclose(p["lr"], expected_lr)
