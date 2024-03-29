from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.optimizers.ranger import Ranger
from src.optimizers.rangerlars import RangerLars

STR_TO_SCHEDULER = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,
}


class PLModule(pl.LightningModule):
    def __init__(
        self,
        checkpoint: Path,
        tok_emb_dim: int,
        mtl_loss_factor: float = 0.1,
        learning_rate: float = 0.00001,
        optimizer: str = "radam",
        no_decay_params: Optional[list] = None,
        weight_decay: float = 0.01,
        scheduler: str = "linear",
        num_training_steps: int = 300_000,
        num_warmup_steps: int = 0,
        *args,
        **kwargs,
    ):
        if no_decay_params is None:
            no_decay_params = ["bias", "LayerNorm.weight"]
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        self.validation_step_outputs: list = []
        if self.qa_model.get_input_embeddings().num_embeddings != tok_emb_dim:
            self.qa_model.resize_token_embeddings(
                tok_emb_dim,
                pad_to_multiple_of=16,
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        start_positions=None,
        end_positions=None,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        outputs = self.qa_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=True,
        )

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        output_dict = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_predictions": torch.argmax(start_logits, dim=1),
            "end_predictions": torch.argmax(end_logits, dim=1),
        }

        if start_positions is not None:
            output_dict["loss"] = outputs.loss

        return output_dict

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        forward_output = self.forward(**batch)

        loss = forward_output["loss"]
        if batch["dataset_identifier"].startswith("mtl"):
            loss *= self.hparams.mtl_loss_factor
            self.log(
                "train_loss_mtl",
                loss,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch["input_ids"].size(0),
            )
        else:
            self.log(
                "train_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch["input_ids"].size(0),
            )
        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        forward_output = self.forward(**batch)
        loss = forward_output["loss"]

        dataset_identifier = batch["dataset_identifier"]
        self.log(
            f"{dataset_identifier}_val_loss",
            loss,
            logger=True,
            on_epoch=True,
            batch_size=batch["input_ids"].size(0),
        )
        self.log(
            "val_loss",
            loss,
            logger=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
        )

        out = {
            f'{batch["dataset_identifier"]}_val_loss': loss,
            "dataset_identifier": dataset_identifier,
            "start_predictions": forward_output["start_predictions"],
            "end_predictions": forward_output["end_predictions"],
            "start_positions": batch["start_positions"],
            "end_positions": batch["end_positions"],
        }

        self.validation_step_outputs.append(out)

        return out

    def on_validation_epoch_end(self) -> None:
        total_val_samples = 0
        correct_start_predictions = 0
        correct_end_predictions = 0
        correct_predictions = 0
        in_bound_start_predictions = 0
        in_bound_end_predictions = 0
        in_bound_predictions = 0

        for val_batch_out in self.validation_step_outputs:
            total_val_batch_samples = val_batch_out["start_predictions"].size(0)
            correct_start_predictions_batch = torch.eq(
                val_batch_out["start_predictions"],
                val_batch_out["start_positions"],
            )
            correct_end_predictions_batch = torch.eq(
                val_batch_out["end_predictions"],
                val_batch_out["end_positions"],
            )
            correct_predictions_batch = torch.bitwise_and(
                correct_start_predictions_batch,
                correct_end_predictions_batch,
            )
            in_bound_start_predictions_batch = torch.bitwise_and(
                val_batch_out["start_predictions"] >= val_batch_out["start_positions"],
                val_batch_out["start_predictions"] <= val_batch_out["end_positions"],
            )
            in_bound_end_predictions_batch = torch.bitwise_and(
                val_batch_out["end_predictions"] >= val_batch_out["start_positions"],
                val_batch_out["end_predictions"] <= val_batch_out["end_positions"],
            )
            in_bound_predictions_batch = torch.bitwise_and(
                in_bound_start_predictions_batch,
                in_bound_end_predictions_batch,
            )

            total_val_samples += total_val_batch_samples
            correct_start_predictions += torch.sum(correct_start_predictions_batch)
            correct_end_predictions += torch.sum(correct_end_predictions_batch)
            correct_predictions += torch.sum(correct_predictions_batch)
            in_bound_start_predictions += torch.sum(in_bound_start_predictions_batch)
            in_bound_end_predictions += torch.sum(in_bound_end_predictions_batch)
            in_bound_predictions += torch.sum(in_bound_predictions_batch)

        prefix = self.validation_step_outputs[-1]["dataset_identifier"]
        self.validation_step_outputs.clear()

        self.log(
            f"{prefix}_correct_start_predictions",
            correct_start_predictions / total_val_samples,
            logger=True,
            batch_size=total_val_samples,
        )
        self.log(
            f"{prefix}_correct_end_predictions",
            correct_end_predictions / total_val_samples,
            logger=True,
            batch_size=total_val_samples,
        )
        self.log(
            f"{prefix}_correct_predictions",
            correct_predictions / total_val_samples,
            logger=True,
            batch_size=total_val_samples,
        )
        self.log(
            f"{prefix}_in_bound_start_predictions",
            in_bound_start_predictions / total_val_samples,
            logger=True,
            batch_size=total_val_samples,
        )
        self.log(
            f"{prefix}_in_bound_end_predictions",
            in_bound_end_predictions / total_val_samples,
            logger=True,
            batch_size=total_val_samples,
        )
        self.log(
            f"{prefix}_in_bound_predictions",
            in_bound_predictions / total_val_samples,
            logger=True,
            batch_size=total_val_samples,
        )

    def configure_optimizers(self):
        no_decay = self.hparams.no_decay_params

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
            )
        elif self.hparams.optimizer == "radam":
            optimizer = torch.optim.RAdam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
            )
        elif self.hparams.optimizer == "ranger":
            optimizer = Ranger(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
            )
        elif self.hparams.optimizer == "rangerlars":
            optimizer = RangerLars(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
            )
        else:
            raise NotImplementedError

        lr_scheduler = STR_TO_SCHEDULER[self.hparams.scheduler](
            optimizer=optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )

        return [optimizer], [{"interval": "step", "scheduler": lr_scheduler}]
