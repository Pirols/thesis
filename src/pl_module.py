import os
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import pytorch_lightning as pl
import torch
from transformers import AutoModelForQuestionAnswering
from transformers import get_constant_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup

from src.optimizers.ranger import Ranger
from src.optimizers.rangerlars import RangerLars


STR_TO_SCHEDULER = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,
}


class Module(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(vars(conf))  # self.hparams = conf
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            conf.transformer_model,
        )
        if getattr(self.hparams, "use_special_tokens", False):
            self.qa_model.resize_token_embeddings(self.hparams.vocab_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        start_positions=None,
        end_positions=None,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

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
        self.log(
            "train_loss",
            loss,
            # on_step=True,
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

        return {
            f'{batch["dataset_identifier"]}_val_loss': loss,
            "dataset_identifier": dataset_identifier,
            "start_predictions": forward_output["start_predictions"],
            "end_predictions": forward_output["end_predictions"],
            "start_positions": batch["start_positions"],
            "end_positions": batch["end_positions"],
        }

    def validation_epoch_end(
        self,
        validation_step_outputs: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:

        if not isinstance(validation_step_outputs, list):
            validation_step_outputs = [validation_step_outputs]

        total_val_samples = 0
        correct_start_predictions = 0
        correct_end_predictions = 0
        correct_predictions = 0
        in_bound_start_predictions = 0
        in_bound_end_predictions = 0
        in_bound_predictions = 0

        for val_batch_out in validation_step_outputs:

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

        prefix = validation_step_outputs[-1]["dataset_identifier"]

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


def get_module(args, tokenizer) -> Module:
    module = Module(args)
    if args.start_from_ckpt:
        starting_module = Module.load_from_checkpoint(args.start_from_ckpt)
        tmp_path = f'/tmp/{args.start_from_ckpt.split("/")[-1]}'
        torch.save(starting_module.state_dict(), tmp_path)
        module.load_state_dict(torch.load(tmp_path))
        os.system(f"rm -rf {tmp_path}")
    elif args.use_special_tokens:
        module.qa_model.resize_token_embeddings(len(tokenizer))
    return module
