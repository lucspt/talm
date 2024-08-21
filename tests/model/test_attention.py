import torch
import pytest
from pytest_mock import MockerFixture

from project_llm.model.attention import CausalSelfAttention

n_head = 4
n_embd = n_head * 3


@pytest.fixture
def c_attn() -> CausalSelfAttention:
    return CausalSelfAttention(n_head=n_head, n_embd=n_embd, dropout=0.1)


def test_raises_if_head_and_embd_sizes_incompatiable() -> None:
    with pytest.raises(ValueError):
        CausalSelfAttention(n_head=3, n_embd=10)


class TestCausalSelfAttention:
    batch_size = 2

    @pytest.fixture
    def mock_inputs(self) -> torch.Tensor:
        return torch.randn(
            self.batch_size,
            8,  # context_len
            n_embd,
        )

    def test_create_heads(self, c_attn: CausalSelfAttention) -> None:
        B, T, C = 32, 8, n_embd
        mock_query_tensor = torch.randn((B, T, C))
        with_heads = c_attn.create_heads(mock_query_tensor, B, T, C, n_head)
        assert with_heads.shape == (B, n_head, T, C // n_head)

    def test_output_shape(
        self, c_attn: CausalSelfAttention, mock_inputs: torch.Tensor
    ) -> None:
        out = c_attn(mock_inputs)
        assert isinstance(out, torch.Tensor)
        assert out.shape == mock_inputs.shape

    @pytest.mark.parametrize("dropout_p", [0.0, 0.1])
    def test_dropout_used_correctly(
        self, dropout_p: float, mock_inputs: torch.Tensor, mocker: MockerFixture
    ) -> None:
        from project_llm.model.attention import nn as nn_spy  # type: ignore

        dropout_spy = mocker.spy(nn_spy, "Dropout")
        c = CausalSelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout_p)
        dropout_spy.assert_called_once_with(dropout_p)

    def test_dropout_off_in_eval_mode(
        self,
        c_attn: CausalSelfAttention,
        mocker: MockerFixture,
        mock_inputs: torch.Tensor,
    ) -> None:
        from project_llm.model.attention import F as F_spy  # type: ignore

        spy = mocker.spy(F_spy, "scaled_dot_product_attention")
        c_attn.eval()
        c_attn(mock_inputs)
        assert spy.call_args.kwargs.get("dropout_p") == 0.0
