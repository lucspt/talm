from typing import Any, Optional

import torch
from tokencoder import Tokenizer

from .types import PathLike
from .resources import Message
from .tokenizer import ChatTokenizer
from .model.model import Model
from .config.model import ModelConfig


class LM:
    @staticmethod
    def build(
        model_path: PathLike, tokenizer_path: PathLike, **model_args: Any
    ) -> "LM":
        """Build a model and tokenizer with the given paths and construct an
        `LM` with the loaded objects.

        Args:
            model_path (PathLike): The model checkpoint path.
            tokenizer_path (PathLike): The path of the tokenizer to use.
            model_args: Keyword arguments to use when creating the `ModelConfig`

        Returns:
            `LM`: The language model
        """
        checkpoint = torch.load(model_path, weights_only=True)
        args = ModelConfig(**checkpoint["config"])
        tokenizer = Tokenizer.from_file(tokenizer_path)
        model = Model(args, vocab_size=tokenizer.n_vocab)
        if "model" not in checkpoint:
            raise Exception(
                f"Could not find `model` in the state dict loaded from `{model_path}`."
            )
        model.load_state_dict(checkpoint["model"])
        return LM(model, tokenizer)

    def __init__(self, model: Model, tokenizer: Tokenizer):
        self.model = model
        self.chat_tokenizer = ChatTokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.model.eval()
        self.stop_tokens = {self.tokenizer.eot_token}

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
        top_p: float = 1.0,
        temperature: Optional[float] = None,
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
        config = self.model.config
        if prompt_len > config.context_len:
            raise Exception(
                f"Context length too long, expected {config.context_len}, got {prompt_len}."
            )
        if max_tokens is None:
            gen_len = config.context_len
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

        for cur_pos in range(prompt_len, gen_len):
            logits = self.model.forward(tokens[:cur_pos])

            probs = torch.softmax(logits[:, -1], dim=-1)

            if temperature:
                probs = probs / temperature

            next_token = self.top_p_sample(probs, p=top_p)
            next_token = next_token.reshape(-1)

            if next_token.item() in self.stop_tokens:
                break

            tokens[:, cur_pos] = next_token

        return tokens.squeeze()[prompt_len:].tolist()

    def chat_completion(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        device: str | None = None,
        top_p: float = 1.0,
        temperature: Optional[float] = None,
    ) -> Message:
        """Perform a chat completion given a list of `Message` objects.

        Args:
            messages (list[Message]): The dialog to generate from.
            max_tokens (int, optional): A maximum number of tokens to generate,
                if this number is greater than the model's context length, it is ignored.
            device (str, optional): The device to move the model to.
            top_p (float): The p value to use when performing top p / nucleus sampling.
            temperature (float, optional): The sampling temperature to use.

        Returns:
            `Message`: The generated response.
        """
        prompt_tokens = self.chat_tokenizer.encode_chat(
            messages, add_generation_prompt=True
        )
        tokens = self.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        return {"role": "assistant", "content": self.tokenizer.decode(tokens)}
