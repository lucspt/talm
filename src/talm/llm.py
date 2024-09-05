from typing import Any, Optional

import torch
from tokencoder import Tokenizer

from .types import PathLike
from .model.model import Model
from .config.model import ModelConfig


class LLM:
    @staticmethod
    def build(
        model_path: PathLike, tokenizer_path: PathLike, **model_args: Any
    ) -> "LLM":
        """Build a model and tokenizer with the given paths and construct an
        `LLM` with the loaded objects.

        Args:
            model_path (PathLike): The model checkpoint path.
            tokenizer_path (PathLike): The path of the tokenizer to use.
            model_args: Keyword arguments to use when creating the `ModelConfig`

        Returns:
            `LLM`: The llm
        """
        args = ModelConfig(**model_args)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        model = Model(args, vocab_size=tokenizer.n_vocab)
        state_dict = torch.load(model_path, weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        else:
            raise Exception(
                f"Could not find `model` in the state dict loaded from `{model_path}`."
            )
        model.load_state_dict(state_dict)
        return LLM(model, tokenizer)

    def __init__(self, model: Model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def top_p_sample(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        """Perform top p sampling over `probs` distribution.

        Args:
            probs (torch.Tensor): The probabilities.
            p (float): The p thershold to use when sampling.

        Returns:
            `torch.Tensor`: The sampled token, one for each batch of `probs`.
        """
        probs_sort, probs_idx = torch.sort(probs)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: Optional[int] = None,
        device: Optional[str] = None,
        top_p: float = 0.5,
        temperature: Optional[float] = 1.0,
    ) -> list[int]:
        """Generate a list of tokens given a list of input tokens.

        Args:
            prompt_tokens (list[int]): The prompt tokens to generate off of.
            max_tokens (int, optional): A maximum number of tokens to generate,
                if this number is greater than the model's context length, it is ignored.
            device (str, optional): The device to move the model to.
            top_p (float): The p value to use when performing top p / nucleus sampling.
            temperature (float, optional): The sampling temperature to use.

        Returns:
            `list[int]`: The generated tokens.
        """
        prompt_len = len(prompt_tokens)
        if max_tokens is None:
            gen_len = self.model.config.context_len
        else:
            gen_len = prompt_len + max_tokens

        tokens = torch.zeros(
            (
                1,
                gen_len,
            ),
            dtype=torch.long,
            device=device,
        )
        tokens[:, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long)
        stop_tokens = {self.tokenizer.eot_token}

        for cur_pos in range(prompt_len, gen_len):
            logits = self.model.forward(tokens[:cur_pos])

            probs = torch.softmax(logits[:, -1], dim=-1)

            if temperature:
                probs = probs / temperature

            next_token = self.top_p_sample(probs, p=top_p)
            next_token = next_token.reshape(-1)

            if next_token.item() in stop_tokens:
                break

            tokens[:, cur_pos] = next_token

        return tokens.squeeze()[prompt_len:].tolist()
