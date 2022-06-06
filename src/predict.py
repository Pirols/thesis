import argparse
from pathlib import Path

import torch
from transformers import AutoModel
from transformers import AutoTokenizer

from src.rainbow import rainbow_file_iter
from src.tokens import OPT_TOKENS


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="microsoft/deberta-v3-large",
    )

    parser.add_argument("--dataset_path", type=str, default="datasets/rainbow")
    parser.add_argument("--dataset_id", type=str, default="physicaliqa")

    parser.add_argument("--output_file", type=str, default="labels.txt")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = AutoModel.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    tokenized_special_tokens = []
    for start_tok, end_tok in OPT_TOKENS[args.dataset_id]:
        tokenized_special_tokens.append(
            (
                tokenizer.encode(start_tok, add_special_tokens=False)[0],
                tokenizer.encode(end_tok, add_special_tokens=False)[0],
            ),
        )

    with open(args.output_file, mode="w", encoding="utf-8") as fd_out:
        for line, _ in rainbow_file_iter(
            Path(args.dataset_path)
            / args.dataset_id
            / ("test." + args.dataset_id + ".csv"),
            OPT_TOKENS[args.dataset_id][-1][-1],
        ):
            inputs = tokenizer(
                line,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"]

            line_start_end_indices = []
            for start_tok, end_tok in tokenized_special_tokens:
                line_start_end_indices.append(
                    (
                        (input_ids.squeeze() == start_tok).nonzero().item(),
                        (input_ids.squeeze() == end_tok).nonzero().item(),
                    ),
                )

            outputs = model(
                input_ids=input_ids,
                return_dict=True,
            )

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_pred = torch.argmax(start_logits, dim=1)
            end_pred = torch.argmax(end_logits, dim=1)

            try:
                idx = line_start_end_indices.index((start_pred, end_pred))
                fd_out.write(idx)
            except ValueError:
                # Running when start and end predictions belong to different sentence
                curr_idx = 0
                cur_tot = -float("inf")

                for i, (start, end) in enumerate(line_start_end_indices):
                    tot = start_logits[start] + end_logits[end]
                    if tot > cur_tot:
                        curr_idx = i
                    cur_tot = tot

                fd_out.write(curr_idx)
