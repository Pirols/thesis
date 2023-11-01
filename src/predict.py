import argparse
import os
from pathlib import Path

from transformers import AutoTokenizer

from src.pl_module import PLModule
from src.rainbow import rainbow_file_iter
from src.tokens import OPT_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--use_fast", action="store_true", default=False)

    parser.add_argument("--data_path", type=str, default="datasets/rainbow")
    parser.add_argument("--dataset", type=str, default="physicaliqa")
    parser.add_argument("--output", type=str, default="experiments/piqa_labels.txt")

    args = parser.parse_args()

    # Use pathlib.Path for paths
    args.checkpoint = Path(args.checkpoint)
    args.data_path = Path(args.data_path)
    args.output_fpath = Path(args.output_fpath)

    return args


def predict(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint.parent.parent / "tokenizer",
        use_fast=args.use_fast,
        add_prefix_space=True,
    )

    model = PLModule.load_from_checkpoint(args.checkpoint)

    tokenized_special_tokens = []
    for start_tok, end_tok in OPT_TOKENS[args.dataset_id]:
        tokenized_special_tokens.append(
            (
                tokenizer.encode(start_tok, add_special_tokens=False)[0],
                tokenizer.encode(end_tok, add_special_tokens=False)[0],
            ),
        )

    with args.output_fpath.open(mode="w", encoding="utf-8") as fd_out:
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
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            try:
                idx = line_start_end_indices.index(
                    (outputs["start_predictions"], outputs["end_predictions"]),
                )
                fd_out.write(idx)
            except ValueError:
                # Running when start and end predictions belong to different sentence
                cur_idx = 0
                cur_tot = -float("inf")

                for i, (start, end) in enumerate(line_start_end_indices):
                    tot = start_logits[start] + end_logits[end]
                    if tot > cur_tot:
                        cur_idx = i
                    cur_tot = tot

                fd_out.write(cur_idx)


if __name__ == "__main__":
    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    predict(parse_args())
