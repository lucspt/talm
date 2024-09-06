import inspect
from pathlib import Path
from argparse import ArgumentParser

import torch
from datasets import load_dataset  # type: ignore
from tokencoder import Tokenizer

from .common import (
    count_params,
    create_optimizer,
    catch_interruption,
    print_training_run,
    get_seeded_generator,
    assert_model_checkpoint_does_not_exist,
)
from ...model import Model
from .trainer import SFTrainer
from ...config import Config
from ...logger import create_logger
from ...tokenizer import ChatTokenizer
from ...data.utils import SFTDataset
from ...config.model import ModelConfig
from ...lr_scheduler import CosineDecayLR
from ...config.training import SFTConfig


def main() -> None:
    paths: list[Path] = []
    try:
        parser = ArgumentParser(
            "sft",
            usage="sft [OPTIONS]",
            description="Perform supervised fine tuning given a model checkpoint",
        )
        parser.add_argument(
            "-m",
            "--model-path",
            dest="model_path",
            required=True,
            type=Path,
            help="The file location of the model checkpoint",
        )
        parser.add_argument(
            "-t",
            "--tokenizer-path",
            dest="tokenizer_path",
            default="tokenizers/base.json",
            type=str,
            help="The file location of the tokenizer to use",
        )
        parser.add_argument(
            "--seed",
            required=False,
            default=None,
            type=int,
            help="An optional seed to set",
        )

        args = parser.parse_args()
        model_path: Path = args.model_path
        tokenizer_path: str = args.tokenizer_path
        seed: int | None = args.seed

        model_checkpoint = torch.load(model_path, weights_only=True)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        model = Model(
            config=ModelConfig(**model_checkpoint["config"]),
            vocab_size=tokenizer.n_vocab,
        )
        model.load_state_dict(model_checkpoint["model"])
        logger = create_logger(__name__)

        sft_config = SFTConfig()
        config = Config(
            **dict(
                (p, getattr(sft_config, p))
                for p in set(inspect.signature(SFTConfig).parameters).intersection(
                    inspect.signature(Config).parameters
                )
            )
        )
        # override the config with sft_config,
        # but some keys in SFTConfig are not in config, so have to find the intersection

        optimizer = create_optimizer(model, config)

        ds = load_dataset(sft_config.dataset_name, name="default")

        model_name = f"{model_path.parent.name}_instruct"

        checkpoint_dir = config.model_dir / model_name

        assert_model_checkpoint_does_not_exist(checkpoint_dir, logger)

        checkpoint_dir.mkdir(exist_ok=True)

        log_file = checkpoint_dir / "logs.txt"

        paths.append(checkpoint_dir)
        paths.append(log_file)

        chat_tokenizer = ChatTokenizer(tokenizer)
        train_ds = SFTDataset(
            ds["train_sft"],
            context_len=model.config.context_len,
            tokenizer=chat_tokenizer,
        )
        val_ds = SFTDataset(
            ds["test_sft"],
            context_len=model.config.context_len,
            tokenizer=chat_tokenizer,
        )

        generator, seed = get_seeded_generator(seed, logger=logger)

        train_dataloader = train_ds.get_dataloader(
            batch_size=sft_config.batch_size, shuffle=True, generator=generator
        )
        val_dataloader = val_ds.get_dataloader(
            batch_size=sft_config.batch_size, shuffle=False, generator=generator
        )

        lr_scheduler = CosineDecayLR(
            optimizer=optimizer,
            max_steps=len(train_dataloader) // 4,  # ~ 550 steps
            min_lr=config.min_lr,
            max_lr=config.max_lr,
            warmup_steps=1,
        )

        trainer = SFTrainer(
            epochs=sft_config.n_epochs,
            model=model,
            model_name=model_name,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            checkpoint_dir=checkpoint_dir,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            log_file=log_file,
            logging_strategy=config.logging_strategy,
            logging_interval=config.logging_interval,
            seed=seed,
            logger=logger,
        )
        n_params = count_params(model)

        print_training_run(
            run_name=model_name,
            n_params=n_params,
            checkpoint_dir=checkpoint_dir,
            config=config,
            log_file=log_file,
        )

        trainer.train()

    except KeyboardInterrupt:
        catch_interruption(paths)
    except Exception as e:
        print(e)
        catch_interruption(paths)
