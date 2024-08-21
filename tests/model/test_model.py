import random
import string

import torch
import pytest

from project_llm.model.model import Model
from project_llm.config.model import ModelConfig

config = ModelConfig()

batch_size = 16
mock_text = "".join(
    random.choice(string.ascii_letters)
    for _ in range(batch_size * config.context_len + 1)
)
encoder = {s: i for i, s in enumerate(set(mock_text))}
decoder = {v: k for k, v in encoder.items()}
vocab_size = len(encoder)


@pytest.fixture()
def model(device: str) -> Model:
    return Model(
        n_head=config.n_head,
        n_embd=config.n_embd,
        context_len=config.context_len,
        vocab_size=vocab_size,
        dropout=config.dropout,
        ln_eps=config.ln_eps,
    ).to(device)


class TestModel:
    B, T, C = batch_size, config.context_len, config.n_embd

    @pytest.fixture(scope="class")
    def mock_inputs_and_labels(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(
            list(encoder[x] for x in mock_text[: self.B * self.T + 1]),
            dtype=torch.long,
        )
        return (
            tokens[:-1].view(self.B, self.T).to(device),
            tokens[1:].view(self.B, self.T).to(device),
        )

    def test_compute_loss(
        self, model: Model, mock_inputs_and_labels: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        inps, lbls = mock_inputs_and_labels
        logits, _ = model(inps)
        loss = model._compute_loss(logits, lbls)
        assert isinstance(loss, torch.Tensor)

    def test_output_shape(
        self, model: Model, mock_inputs_and_labels: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        logits, _ = model(mock_inputs_and_labels[0])
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (self.B, self.T, vocab_size)

    def test_call_with_target_returns_loss(
        self, model: Model, mock_inputs_and_labels: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        logits, loss = model(*mock_inputs_and_labels)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(logits, torch.Tensor)
        assert isinstance(loss.item(), float)

    def test_random_weights_returns_expected_loss(
        self, model: Model, mock_inputs_and_labels: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        inps, lbls = mock_inputs_and_labels
        expected_loss = -torch.log(torch.tensor(1 / vocab_size))
        _, loss = model(inps, lbls)
        for _ in range(3):
            _, loss = model(inps, target=lbls)
            assert torch.allclose(loss, expected_loss, rtol=1), (
                f"Expected loss for model with random weights was {expected_loss.item()}, "
                f"received loss: {loss.item()}."
            )
