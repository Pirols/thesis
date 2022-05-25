import argparse
import os

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader

from src.datasets import DatasetAlternator
from src.datasets import RainbowExtractiveQADataset
from src.datasets import RainbowExtractiveQAIterableDataset
from src.pl_module import get_module
from src.tokenizer import get_tokenizer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transformer_model",
        type=str,
        default="microsoft/deberta-v3-large",
    )
    parser.add_argument("--use_special_tokens", action="store_true", default=False)
    parser.add_argument("--start_from_ckpt", type=str)

    parser.add_argument("--tokens_per_batch", type=int, default=700)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--gradient_acc_steps", type=int, default=20)
    parser.add_argument("--gradient_clipping", type=float, default=10.0)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--num_training_steps", type=int, default=300_000)
    parser.add_argument("--scheduler", type=str, default="linear")

    parser.add_argument("--optimizer", type=str, default="radam")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--no_decay_params",
        default=["bias", "LayerNorm.weight"],
        action="append",
    )

    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--amp_level", type=str, default="O1")

    parser.add_argument("--save_topk", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--run_name", type=str, default="default_name")
    parser.add_argument("--wandb_project", type=str, default="thesis")

    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--datasets_id", action="append")
    parser.add_argument("--infinite_dataset", action="store_true", default=False)
    parser.add_argument("--iterable_dataset", action="store_true", default=False)
    parser.add_argument("--overfit_batches", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=False)

    args = parser.parse_args()
    if args.iterable_dataset > 1.0:
        args.val_check_interval = int(args.val_check_interval)

    if args.overfit_batches > 1.0:
        args.overfit_batches = int(args.overfit_batches)

    return args


def train(args: argparse.Namespace) -> None:

    tokenizer = get_tokenizer(
        args.transformer_model,
        args.use_special_tokens,
        args.datasets_id,
    )
    if args.use_special_tokens:
        args.vocab_size = len(tokenizer)

    train_datasets, validation_datasets = [], []

    data_class = RainbowExtractiveQADataset
    if args.iterable_dataset:
        data_class = RainbowExtractiveQAIterableDataset

    for dataset_id in args.datasets_id:
        train_datasets.append(
            data_class(
                args.data_path + "/rainbow",
                dataset_id,
                tokenizer,
                args.tokens_per_batch,
                is_test=False,
            ),
        )

        validation_datasets.append(
            data_class(
                args.data_path + "/rainbow",
                dataset_id,
                tokenizer,
                args.tokens_per_batch,
                is_test=True,
            ),
        )

    if args.iterable_dataset:
        train_dataset = (
            DatasetAlternator(train_datasets, is_infinite=args.infinite_dataset)
            if len(train_datasets) > 0
            else train_datasets[0]
        )
    else:
        train_dataset = ConcatDataset(train_datasets)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=args.num_workers,
        shuffle=not args.iterable_dataset,  # if dataset is iterable no shuffling
    )
    validation_dataloader = [
        DataLoader(vd, batch_size=None, num_workers=args.num_workers)
        for vd in validation_datasets
    ]

    # Callbacks
    ## WANDB
    wandb_logger = WandbLogger(name=args.run_name, project=args.wandb_project)
    wandb_logger.log_hyperparams(args)
    ## EarlyStopping
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
    )
    ## ModelCheckpoint
    model_checkpoint = ModelCheckpoint(
        dirpath=f"experiments/{args.run_name}",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=args.save_topk,
        save_last=True,
        verbose=True,
        mode="min",
        monitor="val_loss",
    )
    ## Progress bar callback
    progress_bar_cb = TQDMProgressBar(refresh_rate=50)

    # WSD MODULE
    module = get_module(args, tokenizer)

    # TRAINER
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        overfit_batches=args.overfit_batches,
        accumulate_grad_batches=args.gradient_acc_steps,
        gradient_clip_val=args.gradient_clipping,
        logger=wandb_logger,
        callbacks=[early_stopping_cb, progress_bar_cb, model_checkpoint],
        val_check_interval=args.val_check_interval,
        max_steps=args.num_training_steps,
        precision=args.precision,
        # amp_level=args.amp_level,
        deterministic=args.deterministic,
    )

    # FIT
    trainer.fit(
        module,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )

    # save tokenizer if needed
    if args.use_special_tokens:
        tokenizer.save(f"experiments/{args.run_name}/tokenizer_with_st")


if __name__ == "__main__":

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()
    seed_everything(args.seed, workers=True)

    train(args)
