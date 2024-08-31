import torch
import pytest
from pytest_mock import MockerFixture

from talm.model.encoding import PositionalEncoding

n_embd = 32
vocab_size = 4
batch_size = 10
context_len = 4


@pytest.fixture
def inputs() -> torch.Tensor:
    return torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            vocab_size,
        ),
    )


@pytest.fixture()
def pos_enc() -> PositionalEncoding:
    return PositionalEncoding(
        n_embd=n_embd, vocab_size=vocab_size, context_len=context_len
    )


def test_output_shape(pos_enc: PositionalEncoding, inputs: torch.Tensor) -> None:
    out = pos_enc(inputs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch_size, context_len, n_embd)


def test_both_encodings_are_called(
    pos_enc: PositionalEncoding, mocker: MockerFixture, inputs: torch.Tensor
) -> None:
    t_embd = pos_enc.token_embedding_table
    p_embd = pos_enc.position_embedding_table
    t_embd_spy = mocker.spy(t_embd, "forward")
    p_embd_spy = mocker.spy(p_embd, "forward")
    pos_enc(inputs)
    t_embd_spy.assert_called_once()
    p_embd_spy.assert_called_once()
