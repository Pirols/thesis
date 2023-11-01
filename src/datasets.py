import random
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset

from src.raganato import raganato_file_iter
from src.rainbow import ZERO_INDEX_LABEL_MAPPINGS, rainbow_file_iter
from src.tokens import OPT_TOKENS


class RainbowExtractiveQADataset(Dataset):
    def __init__(
        self,
        rainbow_path: Path,
        dataset_name: str,
        tokenizer,
        tokens_per_batch: int,
        aim: str = "train",  # or "validation" or "test"
    ) -> None:
        self.name = dataset_name
        self.tokenizer = tokenizer
        self.tokens_per_batch = tokens_per_batch
        self.aim = aim

        self.limit_length = min(self.tokens_per_batch, self.tokenizer.model_max_length)
        self.zero_index_label = ZERO_INDEX_LABEL_MAPPINGS[self.name]
        self.fname = rainbow_path / self.name / (self.aim + "." + self.name + ".csv")

        self.start_opt_tokens_id: List = []
        self.end_opt_tokens_id: List = []
        for start_tk, end_tk in OPT_TOKENS[self.name]:
            self.start_opt_tokens_id.append(
                self.tokenizer.encode(start_tk, add_special_tokens=False)[0],
            )
            self.end_opt_tokens_id.append(
                self.tokenizer.encode(end_tk, add_special_tokens=False)[0],
            )

        # initialize dataset
        self.data_store: List = []
        self.init_dataset()

    def init_dataset(self) -> None:
        # Reset dataset
        self.data_store: List = []

        # aggregation variable
        current_batch_input_ids: List = []
        current_batch_attention_masks: List = []
        current_batch_start_positions: List = []
        current_batch_end_positions: List = []

        for line, label_idx in rainbow_file_iter(
            self.fname,
            self.tokenizer.decode(self.end_opt_tokens_id[-1]),
        ):
            # tokenize sequence
            tokenized_sequence = self.tokenizer(line, return_tensors="pt")
            encoded_final_sequence = tokenized_sequence["input_ids"].squeeze()

            line_len = encoded_final_sequence.size(0)

            # line too long, just discard it
            if line_len > self.limit_length:
                continue

            # find label start and end positions (after tokenization)
            start_position = (
                (
                    encoded_final_sequence
                    == self.start_opt_tokens_id[label_idx - (not self.zero_index_label)]
                )
                .nonzero()
                .item()
            )
            end_position = (
                (
                    encoded_final_sequence
                    == self.end_opt_tokens_id[label_idx - (not self.zero_index_label)]
                )
                .nonzero()
                .item()
            )

            # batching entries together
            if (
                max(
                    max((be.size(0) for be in current_batch_input_ids), default=0),
                    line_len,
                )
                * (len(current_batch_input_ids) + 1)
                > self.tokens_per_batch
            ):
                self.data_store.append(
                    {
                        "input_ids": pad_sequence(
                            current_batch_input_ids,
                            batch_first=True,
                            padding_value=self.tokenizer.pad_token_id,
                        ),
                        "attention_mask": pad_sequence(
                            current_batch_attention_masks,
                            batch_first=True,
                            padding_value=0,
                        ),
                        "start_positions": torch.tensor(current_batch_start_positions),
                        "end_positions": torch.tensor(current_batch_end_positions),
                        "dataset_identifier": self.name,
                    },
                )
                current_batch_input_ids = []
                current_batch_attention_masks = []
                current_batch_start_positions = []
                current_batch_end_positions = []

            current_batch_input_ids.append(encoded_final_sequence)
            current_batch_attention_masks.append(
                tokenized_sequence["attention_mask"].squeeze(),
            )
            current_batch_start_positions.append(start_position)
            current_batch_end_positions.append(end_position)

        if current_batch_input_ids:
            self.data_store.append(
                {
                    "input_ids": pad_sequence(
                        current_batch_input_ids,
                        batch_first=True,
                        padding_value=self.tokenizer.pad_token_id,
                    ),
                    "attention_mask": pad_sequence(
                        current_batch_attention_masks,
                        batch_first=True,
                        padding_value=0,
                    ),
                    "start_positions": torch.tensor(current_batch_start_positions),
                    "end_positions": torch.tensor(current_batch_end_positions),
                    "dataset_identifier": self.name,
                },
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data_store[idx]

    def __len__(self) -> int:
        return len(self.data_store)


class RainbowExtractiveQAIterableDataset(IterableDataset):
    def __init__(
        self,
        rainbow_path: Path,
        dataset_name: str,
        tokenizer,
        tokens_per_batch: int,
        aim: str = "train",  # or "validation" or "test"
    ) -> None:
        self.name = dataset_name
        self.tokenizer = tokenizer
        self.tokens_per_batch = tokens_per_batch
        self.aim = aim

        self.limit_length = min(self.tokens_per_batch, self.tokenizer.model_max_length)
        self.zero_index_label = ZERO_INDEX_LABEL_MAPPINGS[self.name]
        self.fname = rainbow_path / self.name / (self.aim + "." + self.name + ".csv")

        self.start_opt_tokens_id = []
        self.end_opt_tokens_id = []
        for start_tk, end_tk in OPT_TOKENS[self.name]:
            self.start_opt_tokens_id.append(
                self.tokenizer.encode(start_tk, add_special_tokens=False)[0],
            )
            self.end_opt_tokens_id.append(
                self.tokenizer.encode(end_tk, add_special_tokens=False)[0],
            )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        current_batch_input_ids: List = []
        current_batch_attention_masks: List = []
        current_batch_start_positions: List = []
        current_batch_end_positions: List = []

        for line, label_idx in rainbow_file_iter(
            self.fname,
            self.tokenizer.decode(self.end_opt_tokens_id[-1]),
        ):
            # tokenize sequence
            tokenized_sequence = self.tokenizer(line, return_tensors="pt")
            encoded_final_sequence = tokenized_sequence["input_ids"].squeeze()

            line_len = encoded_final_sequence.size(0)

            # line too long, just discard it
            if line_len > self.limit_length:
                continue

            # find label start and end positions (after tokenization)
            start_position = (
                (
                    encoded_final_sequence
                    == self.start_opt_tokens_id[label_idx - (not self.zero_index_label)]
                )
                .nonzero()
                .item()
            )
            end_position = (
                (
                    encoded_final_sequence
                    == self.end_opt_tokens_id[label_idx - (not self.zero_index_label)]
                )
                .nonzero()
                .item()
            )

            # batching entries together
            if (
                max(
                    max((be.size(0) for be in current_batch_input_ids), default=0),
                    line_len,
                )
                * (len(current_batch_input_ids) + 1)
                > self.tokens_per_batch
            ):
                yield {
                    "input_ids": pad_sequence(
                        current_batch_input_ids,
                        batch_first=True,
                        padding_value=self.tokenizer.pad_token_id,
                    ),
                    "attention_mask": pad_sequence(
                        current_batch_attention_masks,
                        batch_first=True,
                        padding_value=0,
                    ),
                    "start_positions": torch.tensor(current_batch_start_positions),
                    "end_positions": torch.tensor(current_batch_end_positions),
                    "dataset_identifier": self.name,
                }
                current_batch_input_ids = []
                current_batch_attention_masks = []
                current_batch_start_positions = []
                current_batch_end_positions = []

            current_batch_input_ids.append(encoded_final_sequence)
            current_batch_attention_masks.append(
                tokenized_sequence["attention_mask"].squeeze(),
            )
            current_batch_start_positions.append(start_position)
            current_batch_end_positions.append(end_position)

        if current_batch_input_ids:
            yield {
                "input_ids": pad_sequence(
                    current_batch_input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                ),
                "attention_mask": pad_sequence(
                    current_batch_attention_masks,
                    batch_first=True,
                    padding_value=0,
                ),
                "start_positions": torch.tensor(current_batch_start_positions),
                "end_positions": torch.tensor(current_batch_end_positions),
                "dataset_identifier": self.name,
            }


class RaganatoExtractiveQADataset(Dataset):
    def __init__(
        self,
        raganato_path: Path,
        dataset_name: str,
        tokenizer,
        tokens_per_batch: int,
        aim: str = "train",  # or "validation" or "test"
    ):
        self.name = dataset_name
        self.tokenizer = tokenizer
        self.tokens_per_batch = tokens_per_batch
        self.aim = aim

        self.limit_length = min(self.tokens_per_batch, self.tokenizer.model_max_length)
        self.gloss_tokens = ("<g>", "</g>")
        self.start_gloss_token_id = self.tokenizer.encode(
            self.gloss_tokens[0],
            add_special_tokens=False,
        )[0]
        self.end_gloss_token_id = self.tokenizer.encode(
            self.gloss_tokens[1],
            add_special_tokens=False,
        )[0]

        # get fnames
        self.data_fname = str(Path(raganato_path) / "semcor.data.xml")
        self.gold_fname = str(Path(raganato_path) / "semcor.gold.key.txt")

        # initialize dataset
        self.data_store: List = []
        self.init_dataset()

    def init_dataset(self) -> None:
        # Reset dataset
        self.data_store: List = []

        # aggregation variables
        current_batch_input_ids: List = []
        current_batch_attention_masks: List = []
        current_batch_start_positions: List = []
        current_batch_end_positions: List = []

        for line, label_idx in raganato_file_iter(
            self.data_fname,
            self.gold_fname,
            gloss_tokens=self.gloss_tokens,
        ):
            # tokenize sequence
            tokenized_sequence = self.tokenizer(line)
            encoded_final_sequence = tokenized_sequence["input_ids"].squeeze()

            line_len = encoded_final_sequence.size(0)

            # line too long, just discard it
            if line_len > self.limit_length:
                continue

            # find label start and end positions (after tokenization)
            start_position = (
                (
                    encoded_final_sequence
                    == self.start_opt_tokens_id[label_idx - (not self.zero_index_label)]
                )
                .nonzero()
                .item()
            )
            end_position = (
                (
                    encoded_final_sequence
                    == self.end_opt_tokens_id[label_idx - (not self.zero_index_label)]
                )
                .nonzero()
                .item()
            )

            # batching entries together
            if (
                max(
                    max((be.size(0) for be in current_batch_input_ids), default=0),
                    line_len,
                )
                * (len(current_batch_input_ids) + 1)
                > self.tokens_per_batch
            ):
                self.data_store.append(
                    {
                        "input_ids": pad_sequence(
                            current_batch_input_ids,
                            batch_first=True,
                            padding_value=self.tokenizer.pad_token_id,
                        ),
                        "attention_mask": pad_sequence(
                            current_batch_attention_masks,
                            batch_first=True,
                            padding_value=0,
                        ),
                        "start_positions": torch.tensor(current_batch_start_positions),
                        "end_positions": torch.tensor(current_batch_end_positions),
                        "dataset_identifier": self.dataset_id,
                    },
                )
                current_batch_input_ids = []
                current_batch_attention_masks = []
                current_batch_start_positions = []
                current_batch_end_positions = []

            current_batch_input_ids.append(encoded_final_sequence)
            current_batch_attention_masks.append(
                tokenized_sequence["attention_mask"].squeeze(),
            )
            current_batch_start_positions.append(start_position)
            current_batch_end_positions.append(end_position)

        if current_batch_input_ids:
            self.data_store.append(
                {
                    "input_ids": pad_sequence(
                        current_batch_input_ids,
                        batch_first=True,
                        padding_value=self.tokenizer.pad_token_id,
                    ),
                    "attention_mask": pad_sequence(
                        current_batch_attention_masks,
                        batch_first=True,
                        padding_value=0,
                    ),
                    "start_positions": torch.tensor(current_batch_start_positions),
                    "end_positions": torch.tensor(current_batch_end_positions),
                    "dataset_identifier": self.dataset_id,
                },
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data_store[idx]

    def __len__(self) -> int:
        if not self.data_store:
            self.init_dataset()
        return len(self.data_store)


class DatasetAlternator(IterableDataset):
    def __init__(self, datasets: List[IterableDataset], is_infinite: bool = False):
        self.datasets = {dataset.name: dataset for dataset in datasets}
        self.is_infinite = is_infinite

    def __iter__(self):
        ds_iterators = [(name, iter(ds)) for name, ds in self.datasets.items()]
        while ds_iterators:
            name, curr_iter = random.choice(ds_iterators)
            try:
                next_batch = next(curr_iter)
            except StopIteration:
                ds_iterators.remove((name, curr_iter))
                if self.is_infinite:
                    ds_iterators.append((name, iter(self.datasets[name])))
            else:
                yield next_batch
