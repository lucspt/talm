import torch
import pytest
from pytest_mock import MockerFixture

from talm.model.modules import (
    MLP,
    DecoderLayer,
    CausalSelfAttention,
    RotaryPositionalEmbedding,
)


class TestRotaryPositionalEmbedding:
    n_embd = 32
    vocab_size = 4
    batch_size = 10
    context_len = 4
    n_head = 2
    head_dim = n_embd // n_head

    def test_raises_if_dim_not_even(self) -> None:
        with pytest.raises(ValueError):
            RotaryPositionalEmbedding(dim=31, max_seq_len=100)

    @pytest.fixture
    def inputs(self) -> torch.Tensor:
        # mock the ouputs of attention key / query
        # these are applied before we turn each head into a batch dimension
        return torch.randn(
            self.batch_size, self.context_len, self.n_head, self.head_dim
        )

    @pytest.fixture()
    def pos_emb(self) -> RotaryPositionalEmbedding:
        return RotaryPositionalEmbedding(
            dim=self.head_dim, max_seq_len=self.context_len * 2
        )

    def test_output_shape(
        Self, pos_emb: RotaryPositionalEmbedding, inputs: torch.Tensor
    ) -> None:
        out = pos_emb(inputs)
        assert out.shape == inputs.shape

    def test_first_rotation_has_no_affect(
        self, pos_emb: RotaryPositionalEmbedding, inputs: torch.Tensor
    ) -> None:
        out = pos_emb(inputs)
        # (bsz, ctx_len, n_head, head_dim), therefore test the first embedding of each batch
        assert torch.allclose(out[:, 0], inputs[:, 0])

    def test_embeddings_are_rotated(
        self, pos_emb: RotaryPositionalEmbedding, inputs: torch.Tensor
    ) -> None:
        out = pos_emb(inputs)
        assert torch.allclose(out, inputs) == False


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
    n_embd = n_head * 4
    max_seq_len = 8

    @pytest.fixture
    def c_attn(self) -> CausalSelfAttention:
        return CausalSelfAttention(
            n_head=self.n_head,
            n_embd=self.n_embd,
            max_seq_len=self.max_seq_len,
            dropout=0.1,
        )

    def test_raises_if_head_and_embd_sizes_incompatiable(self) -> None:
        with pytest.raises(ValueError):
            CausalSelfAttention(n_head=3, n_embd=10, max_seq_len=2)

    @pytest.fixture
    def mock_inputs(self) -> torch.Tensor:
        return torch.randn(
            self.batch_size,
            8,  # context_len
            self.n_embd,
        )

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
        CausalSelfAttention(
            n_embd=self.n_embd,
            n_head=self.n_head,
            dropout=dropout_p,
            max_seq_len=self.max_seq_len,
        )
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


class TestDecoderLayer:
    n_embd = 32
    n_head = 4
    context_len = 8

    @pytest.fixture()
    def layer(self) -> DecoderLayer:
        return DecoderLayer(
            n_head=self.n_head, n_embd=self.n_embd, max_seq_len=self.context_len * 2
        )

    def test_output_shape(self, layer: DecoderLayer) -> None:
        B, T, C = 4, self.context_len, self.n_embd
        inputs = torch.randn((B, T, C))
        out = layer(inputs)
        assert isinstance(out, torch.Tensor)
        assert out.shape == inputs.shape
