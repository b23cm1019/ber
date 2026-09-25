"""Create the expected one-row-per-source1 submission TSV."""

import argparse
from pathlib import Path

import polars as pl


def generate_submission(source1_parquet, predictions_path, output_path):
    matched = (
        pl.scan_parquet(predictions_path)
        .filter(pl.col("is_match"))
        .group_by("s1_id")
        .agg(pl.col("candidate_id").sort().str.join(",").alias("matched_entity_ids"))
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_parquet(source1_parquet)
        .select(pl.col("entity_id").alias("source1_entity_id"))
        .join(matched, left_on="source1_entity_id", right_on="s1_id", how="left")
        .with_columns(pl.col("matched_entity_ids").fill_null(""))
        .sink_csv(output_path, separator="\t")
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source1_parquet")
    parser.add_argument("predictions_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    generate_submission(args.source1_parquet, args.predictions_path, args.output_path)


if __name__ == "__main__":
    main()
