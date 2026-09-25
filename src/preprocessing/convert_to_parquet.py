"""Convert raw TSV files to preprocessed Parquet exactly once."""

import argparse
from pathlib import Path

from .preprocess_pipeline import preprocess
from src.utils.io import ensure_parent, scan_tsv


def convert_tsv(input_path: str | Path, output_path: str | Path) -> None:
    output_path = ensure_parent(output_path)
    # sink_parquet keeps the input lazy and avoids a dataframe containing a
    # whole 5M-row source in memory.
    preprocess(scan_tsv(input_path)).sink_parquet(output_path, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    convert_tsv(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
