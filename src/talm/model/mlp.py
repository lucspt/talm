from torch import Tensor, nn


class MLP(nn.Module):
    """Feed forward network / multi-layer perceptron."""

    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4, bias=False),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out
