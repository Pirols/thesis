from pathlib import Path
from typing import Iterator
from typing import Tuple


def rainbow_file_iter(fname: Path, end_token: str) -> Iterator[Tuple[str, int]]:
    """Iterates over the samples contained in fname rainbow formatted file."""
    with open(fname, encoding="UTF-8") as fdesc:
        # discard first line
        fdesc.readline()

        # initialize empty string
        curr_line = ""

        # precompute length of final token
        length_end_token = len(end_token)

        for line in fdesc:
            # discard first line (only containing information about the index of the sample)
            if line.startswith('"'):
                continue

            # check whether end_token is in current line
            idx_end_token = line.find(end_token)
            if idx_end_token != -1:
                curr_line += line[: idx_end_token + length_end_token]
                try:
                    yield curr_line, int(line.rstrip()[-2])
                except ValueError:
                    # If parsing test dataset
                    yield curr_line, -1

                # reset empty string
                curr_line = ""
            else:
                # line is not final, just append it
                curr_line += line.rstrip()


ZERO_INDEX_LABEL_MAPPINGS = {
    "anli": False,
    "cosmosqa": True,
    "hellaswag": True,
    "physicaliqa": True,
    "socialiqa": False,
    "winogrande": True,
}
