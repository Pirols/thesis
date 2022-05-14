import os
from collections import defaultdict
from typing import Union

import torch
from transformers import AutoTokenizer

from csd.tokens import ALL_TOKENS
from csd.tokens import OPT_TOKENS


class ExtractiveQATokenizer:
    def __init__(
        self,
        transformer_model: str,
        use_special_tokens: bool,
        dataset_ids: list[str],
    ) -> None:

        self.transformer_model = transformer_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model,
            use_fast=True,
            add_prefix_space=True,
        )

        self.use_special_tokens = use_special_tokens
        if self.use_special_tokens:

            # accumulate special tokens and add them to the tokenizer
            to_add = set()
            for dataset_id in dataset_ids:
                to_add = to_add.union(ALL_TOKENS[dataset_id])
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": list(to_add)},
            )

            self.start_opt_tokens_id = defaultdict(list)
            self.end_opt_tokens_id = defaultdict(list)

            for dataset_id in dataset_ids:
                for start_token, end_token in OPT_TOKENS[dataset_id]:
                    self.start_opt_tokens_id[dataset_id].append(
                        self.tokenizer.encode(start_token, add_special_tokens=False)[0],
                    )
                    self.end_opt_tokens_id[dataset_id].append(
                        self.tokenizer.encode(end_token, add_special_tokens=False)[0],
                    )

            # space token id
            try:
                self.space_token_id = self.tokenizer.encode(
                    " ",
                    add_special_tokens=False,
                )[0]
            except IndexError:
                self.space_token_id = -1

    @property
    def model_max_length(self) -> int:
        return self.tokenizer.model_max_length

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self.tokenizer.save_pretrained(dir_path)

    def decode(self, sequence: Union[int, list[int]]) -> str:
        if isinstance(sequence, int):
            sequence = [sequence]
        return self.tokenizer.decode(sequence)

    def __len__(self) -> int:
        return len(self.tokenizer)

    def __call__(self, sentences: Union[list[str], str]) -> dict[str, torch.LongTensor]:
        return self.tokenizer(sentences, return_tensors="pt")


def get_tokenizer(
    transformer_model: str,
    use_special_tokens: bool,
    datasets: list[str],
) -> ExtractiveQATokenizer:
    return ExtractiveQATokenizer(transformer_model, use_special_tokens, datasets)
