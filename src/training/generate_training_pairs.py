"""Attach ground-truth labels to retrieved candidates with a lazy Polars join."""

import argparse
from pathlib import Path

import polars as pl


def generate_training_pairs(
    candidate_pairs_path: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path,
    hard_negatives_per_positive: int = 5,
) -> Path:
    """Write positives plus the most similar non-matches for each Source1.

    Labels are intentionally created only after retrieval.  This trains the
    reranker on the distribution it will see at inference and prevents a huge
    Cartesian product of all entity tables.
    """
    labels = (
        pl.scan_csv(ground_truth_path, separator="\t")
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").fill_null("").str.split(",").alias("matches"),
        )
        .explode("matches", empty_as_null=True)
        .with_columns(pl.col("matches").str.strip_chars())
        .filter(pl.col("matches") != "")
        .select("s1_id", pl.col("matches").alias("candidate_id"))
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hard_negatives_per_positive < 1:
        raise ValueError("hard_negatives_per_positive must be at least one")
    labeled = (
        pl.scan_parquet(candidate_pairs_path)
        .join(labels, on=["s1_id", "candidate_id"], how="left")
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
        .with_columns(pl.col("label").sum().over("s1_id").alias("positive_count"))
        .with_columns(
            pl.when(pl.col("label") == 0)
            .then(pl.col("retrieval_score").fill_null(-1.0))
            .otherwise(None)
            .rank("ordinal", descending=True)
            .over("s1_id")
            .alias("negative_rank")
        )
        .filter(
            (pl.col("label") == 1)
            | (
                (pl.col("positive_count") > 0)
                & (pl.col("negative_rank") <= pl.col("positive_count") * hard_negatives_per_positive)
            )
        )
        .drop("positive_count", "negative_rank")
    )
    labeled.sink_parquet(output_path, compression="zstd")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_pairs_path")
    parser.add_argument("ground_truth_path")
    parser.add_argument("output_path")
    parser.add_argument("--hard-negatives-per-positive", type=int, default=5)
    args = parser.parse_args()
    generate_training_pairs(args.candidate_pairs_path, args.ground_truth_path, args.output_path, args.hard_negatives_per_positive)


if __name__ == "__main__":
    main()
