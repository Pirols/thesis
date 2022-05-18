from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

from src.rainbow import rainbow_file_iter
from src.rainbow import ZERO_INDEX_LABEL_MAPPINGS
from src.tokenizer import ExtractiveQATokenizer


class RainbowExtractiveQADataset(Dataset):
    def __init__(
        self,
        rainbow_path: str,
        dataset_id: str,
        tokenizer: ExtractiveQATokenizer,
        tokens_per_batch: int,
        is_test: bool = False,
    ) -> None:

        self.dataset_id = dataset_id
        self.zero_index_label = ZERO_INDEX_LABEL_MAPPINGS[dataset_id]
        self.is_test = is_test

        self.tokenizer = tokenizer
        self.tokens_per_batch = tokens_per_batch
        self.start_opt_tokens_id = self.tokenizer.start_opt_tokens_id[self.dataset_id]
        self.end_opt_tokens_id = self.tokenizer.end_opt_tokens_id[self.dataset_id]

        self.fname = (
            Path(rainbow_path)
            / self.dataset_id
            / (("validation" if is_test else "train") + "." + self.dataset_id + ".csv")
        )

        # initialize dataset
        self.data_store = []
        self.init_dataset()

    def init_dataset(self) -> None:
        # Reset dataset
        self.data_store = []

        # aggregation variable
        current_batch_input_ids = []
        current_batch_attention_masks = []
        current_batch_start_positions = []
        current_batch_end_positions = []

        for line, label_idx in rainbow_file_iter(
            self.fname,
            self.tokenizer.decode(self.end_opt_tokens_id[-1]),
        ):

            # tokenize sequence
            tokenized_sequence = self.tokenizer(line)
            encoded_final_sequence = tokenized_sequence["input_ids"].squeeze()

            line_len = encoded_final_sequence.size(0)

            # line too long, just discard it
            if (
                line_len > self.tokenizer.model_max_length
                or line_len > self.tokens_per_batch
            ):
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


class RainbowExtractiveQAIterableDataset(IterableDataset):
    def __init__(
        self,
        rainbow_path: str,
        dataset_id: str,
        tokenizer: ExtractiveQATokenizer,
        tokens_per_batch: int,
        is_test: bool = False,
    ) -> None:

        # initialize dataset
        self.data_store = []
        self.init_dataset()
        self.dataset_id = dataset_id
        self.zero_index_label = ZERO_INDEX_LABEL_MAPPINGS[dataset_id]
        self.is_test = is_test

        self.tokenizer = tokenizer
        self.tokens_per_batch = tokens_per_batch
        self.start_opt_tokens_id = self.tokenizer.start_opt_tokens_id[self.dataset_id]
        self.end_opt_tokens_id = self.tokenizer.end_opt_tokens_id[self.dataset_id]

        self.fname = (
            Path(rainbow_path)
            / self.dataset_id
            / (("validation" if is_test else "train") + "." + self.dataset_id + ".csv")
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # aggregation variable
        current_batch_input_ids = []
        current_batch_attention_masks = []
        current_batch_start_positions = []
        current_batch_end_positions = []

        for line, label_idx in rainbow_file_iter(
            self.fname,
            self.tokenizer.decode(self.end_opt_tokens_id[-1]),
        ):

            # tokenize sequence
            tokenized_sequence = self.tokenizer(line)
            encoded_final_sequence = tokenized_sequence["input_ids"].squeeze()

            line_len = encoded_final_sequence.size(0)

            # line too long, just discard it
            if (
                line_len > self.tokenizer.model_max_length
                or line_len > self.tokens_per_batch
            ):
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
                    "dataset_identifier": self.dataset_id,
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
                "dataset_identifier": self.dataset_id,
            }


class DatasetAlternator(IterableDataset):
    def __init__(self, datasets: List[IterableDataset], is_infinite: bool = False):
        self.datasets = {dataset.dataset_id: dataset for dataset in datasets}
        self.is_infinite = is_infinite

    def __iter__(self):

        datasets_iterable = {
            dataset.dataset_id: [dataset.__iter__(), False]
            for dataset in self.datasets.values()
        }

        while not all([x[1] for x in datasets_iterable.values()]):
            for dataset_name, (dataset_iterator, is_ended) in datasets_iterable.items():
                if not is_ended:
                    next_batch = next(dataset_iterator, None)
                    if next_batch is None:
                        if not self.is_infinite:
                            datasets_iterable[dataset_name][-1] = True
                        else:
                            datasets_iterable[dataset_name][0] = self.datasets[
                                dataset_name
                            ].__iter__()
                    else:
                        yield next_batch
