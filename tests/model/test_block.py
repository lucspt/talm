import torch
import pytest

from talm.model.block import DecoderBlock

n_head = 4
n_embd = 12
context_len = 8


@pytest.fixture()
def block() -> DecoderBlock:
    return DecoderBlock(n_head=n_head, n_embd=n_embd)


def test_output_shape(block: DecoderBlock) -> None:
    B, T, C = 4, context_len, n_embd
    inputs = torch.randn((B, T, C))
    out = block(inputs)
    assert isinstance(out, torch.Tensor)
    assert out.shape == inputs.shape
