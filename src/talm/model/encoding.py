from torch import Tensor, nn, arange


class PositionalEncoding(nn.Module):
    """Positional encoding that creates embedding tables
    for both positional and informational encoding.
    """

    def __init__(self, n_embd: int, vocab_size: int, context_len: int):
        """Initialize a PositionalEncoding.

        Args:
            n_embd (int): The number of embedding dims for the embedding tables.
            vocab_size (int): The total number of vocab tokens to create
                a token embedding table with
            context_len (int): The length of the context / number of positions to encode
                through a position embedding table.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_len, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        tx = self.token_embedding_table(x)  # (B, seq_len, n_embd)
        posx = self.position_embedding_table(
            arange(x.size(1), device=x.device)
        )  # (B, n_embd)
        out: Tensor = tx + posx
        return out
