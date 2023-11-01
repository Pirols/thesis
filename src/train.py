import argparse
import os
from pathlib import Path
from typing import List, Tuple

import nltk
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from src.datasets import (
    DatasetAlternator,
    RainbowExtractiveQADataset,
    RainbowExtractiveQAIterableDataset,
)
from src.pl_module import PLModule
from src.tokens import ALL_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="microsoft/deberta-v3-base",
    )
    parser.add_argument("--use_fast", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)

    parser.add_argument("--tokens_per_batch", type=int, default=700)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--gradient_acc_steps", type=int, default=20)
    parser.add_argument("--gradient_clipping", type=float, default=10.0)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--num_training_steps", type=int, default=300_000)
    parser.add_argument("--scheduler", type=str, default="linear")

    parser.add_argument("--optimizer", type=str, default="radam")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--no_decay_params",
        default=["bias", "LayerNorm.weight"],
        action="append",
    )

    parser.add_argument("--save_topk", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--run_name", type=str, default="default_name")
    parser.add_argument("--wandb_project", type=str, default="thesis")

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")

    parser.add_argument("--data_path", type=str, default="datasets")
    parser.add_argument("--datasets", action="append")
    parser.add_argument("--infinite_dataset", action="store_true", default=False)
    parser.add_argument("--iterable_dataset", action="store_true", default=False)
    parser.add_argument("--overfit_batches", type=float, default=0.0)

    parser.add_argument("--wsd", action="store_true", default=False)
    parser.add_argument("--mtl_loss_factor", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=False)

    args = parser.parse_args()

    if args.iterable_dataset:
        args.val_check_interval = int(args.val_check_interval)

    if args.overfit_batches > 1.0:
        args.overfit_batches = int(args.overfit_batches)

    if args.resume:
        args.checkpoint = Path(args.checkpoint)
    args.data_path = Path(args.data_path)

    return args


def get_tokenizer(args: argparse.Namespace) -> AutoTokenizer:
    if args.resume:
        return AutoTokenizer.from_pretrained(
            args.checkpoint / "tokenizer",
            use_fast=args.use_fast,
            add_prefix_space=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=args.use_fast,
        add_prefix_space=True,
    )

    # add special tokens for the involved datasets
    additional_special_tokens = []
    for dataset in args.datasets:
        additional_special_tokens.extend(ALL_TOKENS[dataset])
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens},
    )

    # save tokenizer
    tokenizer_path = Path(f"experiments/{args.run_name}/tokenizer")
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)

    return tokenizer


def get_dataloaders(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
) -> Tuple[DataLoader, List[DataLoader]]:
    rainbow_data_class = (
        RainbowExtractiveQAIterableDataset
        if args.iterable_dataset
        else RainbowExtractiveQADataset
    )

    train_datasets, validation_datasets = [], []
    for dataset in args.datasets:
        train_datasets.append(
            rainbow_data_class(
                args.data_path / "rainbow",
                dataset,
                tokenizer,
                args.tokens_per_batch,
            ),
        )

        validation_datasets.append(
            rainbow_data_class(
                args.data_path / "rainbow",
                dataset,
                tokenizer,
                args.tokens_per_batch,
                aim="validation",
            ),
        )

    train_dataset = (
        train_datasets[0]
        if len(train_datasets) == 1
        else (
            DatasetAlternator(train_datasets, is_infinite=args.infinite_dataset)
            if args.iterable_dataset
            else ConcatDataset(train_datasets)
        )
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=args.num_workers,
        shuffle=not args.iterable_dataset,
    )
    validation_dataloader = [
        DataLoader(vd, batch_size=None, num_workers=args.num_workers)
        for vd in validation_datasets
    ]

    return train_dataloader, validation_dataloader


def get_model(args: argparse.Namespace, tokenizer_size: int) -> PLModule:
    if args.resume:
        return PLModule.load_from_checkpoint(args.checkpoint / "ckpts" / "latest.ckpt")

    return PLModule(
        args.checkpoint,
        tokenizer_size,
        args.mtl_loss_factor,
    )


def train(args: argparse.Namespace) -> None:
    tokenizer = get_tokenizer(args)

    train_dataloader, validation_dataloader = get_dataloaders(args, tokenizer)

    model = get_model(args, len(tokenizer))

    # --- Callbacks
    wandb_logger = WandbLogger(name=args.run_name, project=args.wandb_project)
    wandb_logger.log_hyperparams(args)
    wandb_logger.watch(model, log="all")
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
    )
    progress_bar_cb = TQDMProgressBar(refresh_rate=50)
    model_checkpoint = ModelCheckpoint(
        dirpath=f"experiments/{args.run_name}/ckpts",
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=args.save_topk,
        verbose=True,
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        overfit_batches=args.overfit_batches,
        accumulate_grad_batches=args.gradient_acc_steps,
        gradient_clip_val=args.gradient_clipping,
        logger=wandb_logger,
        callbacks=[early_stopping_cb, progress_bar_cb, model_checkpoint],
        val_check_interval=args.val_check_interval,
        max_steps=args.num_training_steps,
        precision=args.precision,
        deterministic=args.deterministic,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
        ckpt_path=args.checkpoint / "ckpts" / "latest.ckpt" if args.resume else None,
    )

    wandb_logger.experiment.unwatch(model)


if __name__ == "__main__":
    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()
    seed_everything(args.seed, workers=True)

    if args.wsd:
        nltk_data_path = "nltk_data"
        nltk.data.path.append(nltk_data_path)
        nltk.download("wordnet", download_dir=nltk_data_path)
        nltk.download("omw-1.4", download_dir=nltk_data_path)

    train(args)
