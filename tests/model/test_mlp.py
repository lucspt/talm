import torch
import pytest
from pytest_mock import MockerFixture

from talm.model.mlp import MLP


def test_output_shape() -> None:
    n_embd = 64
    mlp = MLP(n_embd=n_embd)
    out = mlp(torch.randn((2, n_embd)))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, n_embd)


@pytest.mark.parametrize("dropout_p", [0.1, 0.0])
def test_dropout_is_used_correctly(dropout_p: float, mocker: MockerFixture) -> None:
    from talm.model.mlp import nn as nn_spy  # type: ignore

    spy = mocker.spy(nn_spy, "Dropout")
    MLP(n_embd=64, dropout=dropout_p)
    spy.assert_called_once_with(dropout_p)
