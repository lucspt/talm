import sys
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import asdict

from tokencoder import Tokenizer

from ...data import ShardedDataLoader
from .common import (
    count_params,
    create_optimizer,
    catch_interruption,
    print_training_run,
    get_seeded_generator,
    assert_model_checkpoint_does_not_exist,
)
from ...model import Model
from .trainer import Trainer
from ...config import Config
from ...logger import create_logger
from ...config.model import ModelConfig
from ...lr_scheduler import CosineDecayLR


def main() -> None:
    try:
        logger = create_logger(__name__)
        paths: list[Path] = []
        model_config = ModelConfig()  # store a plain model config for saving
        config = Config(**asdict(model_config))
        parser = ArgumentParser(
            "train_model",
            description="Pre train a language model.",
            usage="train_model <MODEL_NAME> <DATA_DIR> [options]",
        )
        parser.add_argument(
            "-n", "--model-name", dest="model_name", required=True, type=str
        )
        parser.add_argument(
            "-d", "--data-dir", dest="data_dir", required=True, type=str
        )
        parser.add_argument(
            "-t",
            "--tokenizer-path",
            dest="tokenizer_path",
            required=False,
            default=config.tokenizer_dir / "base.json",
            type=str,
        )

        parser.add_argument(
            "-s",
            "--seed",
            dest="seed",
            required=False,
            type=int,
            default=None,
        )
        args = parser.parse_args()
        model_name: str = args.model_name
        data_dir = args.data_dir
        seed: int | None = args.seed
        tokenizer_path: Path = Path(args.tokenizer_path)

        if not tokenizer_path.exists():
            logger.error(
                f"The tokenizer path {tokenizer_path} does not exist. "
                "Please specify a valid tokenizer path"
            )
            sys.exit(1)

        tokenizer = Tokenizer.from_file(tokenizer_path)

        generator, seed = get_seeded_generator(seed, logger)

        train_dataloader = ShardedDataLoader(
            split="train",
            batch_size=config.batch_size,
            context_len=config.context_len,
            dirname=data_dir,
            shuffle=True,
            generator=generator,
        )

        val_dataloader = ShardedDataLoader(
            split="val",
            batch_size=config.batch_size,
            context_len=config.context_len,
            dirname=data_dir,
            shuffle=False,
        )

        model = Model(config=model_config, vocab_size=tokenizer.n_vocab)

        optimizer = create_optimizer(model=model, config=config)

        config.model_dir.mkdir(exist_ok=True)
        checkpoint_dir = config.model_dir / model_name

        assert_model_checkpoint_does_not_exist(checkpoint_dir, logger)

        checkpoint_dir.mkdir(exist_ok=True)
        paths.append(checkpoint_dir)

        log_file = checkpoint_dir / f"logs.txt"

        paths.append(log_file)

        lr_max_steps = len(train_dataloader)
        lr_warmup_steps = int(1e7) // (
            config.batch_size * config.context_len
        )  # warmup first 10m tokens

        lr_scheduler = CosineDecayLR(
            optimizer=optimizer,
            warmup_steps=lr_warmup_steps,
            max_steps=lr_max_steps,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
        )

        trainer = Trainer(
            epochs=config.n_epochs,
            log_file=log_file,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            lr_scheduler=lr_scheduler,
            logger=logger,
            seed=seed,
            logging_strategy=config.logging_strategy,
            logging_interval=config.logging_interval,
        )

        n_params = count_params(model)

        print_training_run(model_name, n_params, checkpoint_dir, log_file, config)

        trainer.train()

    except KeyboardInterrupt:
        catch_interruption(paths)
    except Exception as e:
        print(e)
        catch_interruption(paths)
