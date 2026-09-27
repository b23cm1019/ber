"""Categorize why ground-truth matches are missing from final candidate pairs.

Run this AFTER regenerating candidate_pairs.parquet with your new top_k /
relaxed blocker / address channel (changes 1, 4, 5), so the audit reflects
your improved retrieval, not the old 50/tight-blocker baseline.

Usage:
    python -m src.evaluation.audit_misses --config configs/config.yaml

Writes a category breakdown to stdout and a sampled, labeled parquet of the
misses to data/candidates/miss_audit_sample.parquet for manual spot-checking.
"""

import argparse
import re
from pathlib import Path

import duckdb
import polars as pl
from rapidfuzz import fuzz

from src.utils.paths import project_path

_ADDR_ABBR_PAIRS = [
    ("rd", "road"), ("st", "street"), ("ave", "avenue"), ("apt", "apartment"),
    ("blvd", "boulevard"), ("dr", "drive"), ("ln", "lane"), ("flr", "floor"),
]
_NON_ASCII = re.compile(r"[^\x00-\x7F]")


def _ground_truth_pairs(path) -> pl.DataFrame:
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
        .collect()
    )


def _missed_pairs(gt: pl.DataFrame, candidate_pairs_path) -> pl.DataFrame:
    candidates = pl.scan_parquet(candidate_pairs_path).select("s1_id", "candidate_id").unique().collect()
    return gt.join(candidates, on=["s1_id", "candidate_id"], how="anti")


def _categorize(row: dict) -> list[str]:
    flags = []
    left_name, right_name = row["core_name_l"] or "", row["core_name_r"] or ""
    left_addr, right_addr = row["clean_address_l"] or "", row["clean_address_r"] or ""

    name_ratio = fuzz.ratio(left_name, right_name)
    name_token_sort = fuzz.token_sort_ratio(left_name, right_name)
    addr_ratio = fuzz.ratio(left_addr, right_addr)
    left_numbers, right_numbers = set(row["numbers_l"] or []), set(row["numbers_r"] or [])
    numeric_overlap = bool(left_numbers & right_numbers)

    # Address-driven: names disagree strongly but the address clearly agrees.
    if name_ratio < 50 and (addr_ratio > 75 or numeric_overlap):
        flags.append("address_driven")

    # Cross-script: raw (pre-normalization) name has non-ASCII on either side.
    if _NON_ASCII.search(row["business_name_l"] or "") or _NON_ASCII.search(row["business_name_r"] or ""):
        flags.append("cross_script")

    # Token reorder: sorting tokens fixes most of the gap that plain ratio sees.
    if name_token_sort - name_ratio > 20:
        flags.append("token_reorder")

    # Abbreviation: one side has the short form, the other the expanded form.
    for short, long in _ADDR_ABBR_PAIRS:
        pat_short, pat_long = rf"\b{short}\b", rf"\b{long}\b"
        if (re.search(pat_short, left_addr) and re.search(pat_long, right_addr)) or (
            re.search(pat_short, right_addr) and re.search(pat_long, left_addr)
        ):
            flags.append("abbreviation")
            break

    # Legal suffix: one side kept a legal form the other stripped/lacks, hurting name ratio.
    if (bool(row["legal_form_l"]) != bool(row["legal_form_r"])) and name_ratio < 85:
        flags.append("legal_suffix")

    if not flags:
        flags.append("unexplained")
    return flags


def audit_misses(
    candidate_pairs_path, ground_truth_path, source1_path, source2_path, source3_path,
    sample_size: int = 1000, seed: int = 42,
) -> pl.DataFrame:
    gt = _ground_truth_pairs(ground_truth_path)
    missed = _missed_pairs(gt, candidate_pairs_path)
    total_gt = len(gt)
    total_missed = len(missed)
    if total_missed == 0:
        print("No missed ground-truth pairs. Retrieval recall is already 100% on this GT set.")
        return pl.DataFrame()
    sample = missed.sample(n=min(sample_size, total_missed), seed=seed, shuffle=True)

    con = duckdb.connect()
    all_ids = list(set(sample["s1_id"].to_list()) | set(sample["candidate_id"].to_list()))
    con.register("ids", pl.DataFrame({"entity_id": all_ids}))
    cols = "entity_id, business_name, business_address, core_name, clean_address, country, legal_form, numbers, phonetic_key, first_numeric_token"
    entities = pl.from_arrow(con.execute(f"""
        SELECT e.* FROM ids i
        JOIN (
            SELECT {cols} FROM read_parquet(?) UNION ALL
            SELECT {cols} FROM read_parquet(?) UNION ALL
            SELECT {cols} FROM read_parquet(?)
        ) e ON e.entity_id = i.entity_id
    """, [str(source1_path), str(source2_path), str(source3_path)]).to_arrow_table())
    con.close()

    left = entities.rename({c: f"{c}_l" for c in entities.columns if c != "entity_id"}).rename({"entity_id": "s1_id"})
    right = entities.rename({c: f"{c}_r" for c in entities.columns if c != "entity_id"}).rename({"entity_id": "candidate_id"})
    joined = sample.join(left, on="s1_id", how="left").join(right, on="candidate_id", how="left")

    records = joined.to_dicts()
    all_flags: list[str] = []
    labeled_rows = []
    for row in records:
        flags = _categorize(row)
        all_flags.extend(flags)
        labeled_rows.append({**row, "categories": ",".join(flags)})

    counts = pl.DataFrame({"flag": all_flags}).group_by("flag").len().sort("len", descending=True)
    print(f"\nGround-truth pairs total:  {total_gt}")
    print(f"Missed (not retrieved):    {total_missed}  ({total_missed / total_gt:.1%} of GT)")
    print(f"Audited sample:            {len(sample)}\n")
    print("Miss category breakdown (a miss can have multiple flags):")
    for flag, n in zip(counts["flag"], counts["len"]):
        print(f"  {flag:<18} {n:>5}  ({n / len(sample):.1%} of sampled misses)")

    return pl.DataFrame(labeled_rows)


def main() -> None:
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()
    with project_path(args.config).open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    paths = {name: project_path(value) for name, value in cfg["paths"].items()}
    labeled = audit_misses(
        paths["candidate_pairs"], paths["train_gt"],
        paths["parquet_dir"] / "source1.parquet",
        paths["parquet_dir"] / "source2.parquet",
        paths["parquet_dir"] / "source3.parquet",
        args.sample_size,
    )
    if len(labeled):
        out = paths["candidate_pairs"].parent / "miss_audit_sample.parquet"
        labeled.write_parquet(out)
        print(f"\nLabeled sample written to {out}")


if __name__ == "__main__":
    main()
