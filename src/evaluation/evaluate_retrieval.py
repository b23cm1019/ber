"""Measure whether ground-truth entity matches survive candidate retrieval."""

import argparse
from pathlib import Path

import polars as pl


def _ground_truth_pairs(path: str | Path) -> pl.LazyFrame:
    return (
        pl.scan_csv(path, separator="\t")
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").fill_null("").str.split(",").alias("matches"),
        )
        .explode("matches", empty_as_null=True)
        .with_columns(pl.col("matches").str.strip_chars())
        .filter(pl.col("matches") != "")
        .select("s1_id", pl.col("matches").alias("candidate_id"))
        .unique()
    )


def evaluate_retrieval(candidate_path: str | Path, ground_truth_path: str | Path, cutoffs=(20, 50, 100)) -> dict[int, float]:
    """Return pair-level Recall@K for candidates ranked independently per target.

    A ground-truth pair counts as recalled if it occurs within its target
    source's Top-K list.  This is the upper bound on downstream reranker
    recall, so it must be measured before LightGBM training.
    """
    candidates = pl.scan_parquet(candidate_path)
    columns = set(candidates.collect_schema().names())
    if "retrieval_rank" not in columns:
        raise ValueError("candidate pairs have no retrieval_rank; regenerate them with the current retriever")
    truth = _ground_truth_pairs(ground_truth_path)
    total = truth.select(pl.len().alias("count")).collect().item()
    if total == 0:
        raise ValueError("Ground truth contains no match pairs")

    results = {}
    for cutoff in sorted(set(cutoffs)):
        recalled = (
            truth.join(
                candidates.filter(pl.col("retrieval_rank") <= cutoff).select("s1_id", "candidate_id").unique(),
                on=["s1_id", "candidate_id"], how="inner",
            )
            .select(pl.len().alias("count"))
            .collect()
            .item()
        )
        results[int(cutoff)] = recalled / total
    return results


def print_retrieval_report(results: dict[int, float]) -> None:
    for cutoff, recall in sorted(results.items()):
        print(f"Recall@{cutoff} = {recall:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_pairs")
    parser.add_argument("ground_truth")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[20, 50, 100])
    args = parser.parse_args()
    print_retrieval_report(evaluate_retrieval(args.candidate_pairs, args.ground_truth, args.cutoffs))


if __name__ == "__main__":
    main()
