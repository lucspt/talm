import torch
import pytest
from pytest_mock import MockerFixture

from talm.model.modules import (
    MLP,
    DecoderBlock,
    PositionalEncoding,
    CausalSelfAttention,
)


class TestPositionalEncoding:
    n_embd = 32
    vocab_size = 4
    batch_size = 10
    context_len = 4

    @pytest.fixture
    def inputs(
        self,
    ) -> torch.Tensor:
        return torch.randint(
            0,
            self.vocab_size,
            (
                self.batch_size,
                self.vocab_size,
            ),
        )

    @pytest.fixture()
    def pos_enc(self) -> PositionalEncoding:
        return PositionalEncoding(
            n_embd=self.n_embd, vocab_size=self.vocab_size, context_len=self.context_len
        )

    def test_output_shape(
        self, pos_enc: PositionalEncoding, inputs: torch.Tensor
    ) -> None:
        out = pos_enc(inputs)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (self.batch_size, self.context_len, self.n_embd)

    def test_both_encodings_are_called(
        self, pos_enc: PositionalEncoding, mocker: MockerFixture, inputs: torch.Tensor
    ) -> None:
        t_embd = pos_enc.token_embedding_table
        p_embd = pos_enc.position_embedding_table
        t_embd_spy = mocker.spy(t_embd, "forward")
        p_embd_spy = mocker.spy(p_embd, "forward")
        pos_enc(inputs)
        t_embd_spy.assert_called_once()
        p_embd_spy.assert_called_once()


class TestMLP:
    def test_output_shape(self) -> None:
        n_embd = 64
        mlp = MLP(n_embd=n_embd)
        out = mlp(torch.randn((2, n_embd)))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, n_embd)

    @pytest.mark.parametrize("dropout_p", [0.1, 0.0])
    def test_dropout_is_used_correctly(
        self, dropout_p: float, mocker: MockerFixture
    ) -> None:
        from talm.model.modules import nn as nn_spy  # type: ignore

        spy = mocker.spy(nn_spy, "Dropout")
        MLP(n_embd=64, dropout=dropout_p)
        spy.assert_called_once_with(dropout_p)


class TestCausalSelfAttention:
    batch_size = 2
    n_head = 4
    n_embd = n_head * 3

    @pytest.fixture
    def c_attn(self) -> CausalSelfAttention:
        return CausalSelfAttention(n_head=self.n_head, n_embd=self.n_embd, dropout=0.1)

    def test_raises_if_head_and_embd_sizes_incompatiable(self) -> None:
        with pytest.raises(ValueError):
            CausalSelfAttention(n_head=3, n_embd=10)

    @pytest.fixture
    def mock_inputs(self) -> torch.Tensor:
        return torch.randn(
            self.batch_size,
            8,  # context_len
            self.n_embd,
        )

    def test_create_heads(self, c_attn: CausalSelfAttention) -> None:
        B, T, C = 32, 8, self.n_embd
        mock_query_tensor = torch.randn((B, T, C))
        with_heads = c_attn.create_heads(mock_query_tensor, B, T, C, self.n_head)
        assert with_heads.shape == (B, self.n_head, T, C // self.n_head)

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
        from talm.model.modules import nn as nn_spy  # type: ignore

        dropout_spy = mocker.spy(nn_spy, "Dropout")
        CausalSelfAttention(n_embd=self.n_embd, n_head=self.n_head, dropout=dropout_p)
        dropout_spy.assert_called_once_with(dropout_p)

    def test_dropout_off_in_eval_mode(
        self,
        c_attn: CausalSelfAttention,
        mocker: MockerFixture,
        mock_inputs: torch.Tensor,
    ) -> None:
        from talm.model.modules import F as F_spy  # type: ignore

        spy = mocker.spy(F_spy, "scaled_dot_product_attention")
        c_attn.eval()
        c_attn(mock_inputs)
        assert spy.call_args.kwargs.get("dropout_p") == 0.0


class TestDecoderBlock:
    n_head = 4
    n_embd = 12
    context_len = 8

    @pytest.fixture()
    def block(self) -> DecoderBlock:
        return DecoderBlock(n_head=self.n_head, n_embd=self.n_embd)

    def test_output_shape(self, block: DecoderBlock) -> None:
        B, T, C = 4, self.context_len, self.n_embd
        inputs = torch.randn((B, T, C))
        out = block(inputs)
        assert isinstance(out, torch.Tensor)
        assert out.shape == inputs.shape
