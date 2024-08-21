import torch
import pytest


@pytest.fixture(scope="module")
def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
