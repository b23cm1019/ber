"""Join retrieved pairs to entity data and write bounded reranker feature files."""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from rapidfuzz import fuzz
import jellyfish


FEATURE_COLUMNS = [
    "name_ratio", "name_token_sort", "name_token_set", "name_jaro",
    "addr_ratio", "addr_token_sort", "numeric_jaccard", "country_match",
    "legal_match", "embedding_cosine",
    # Containment/overlap coefficient (len(intersection)/min(len(A),len(B))),
    # unlike Jaccard this stays high when one side is a short name/address that is a
    # near-subset of a longer one, e.g. "ABC Technologies" vs "ABC Technologies Pvt Ltd Mumbai".
    "name_overlap_coef", "addr_overlap_coef",
    # Character-trigram Jaccard, robust to spacing/concatenation differences that
    # token-based measures miss, e.g. "shreebalaji" vs "shree balaji", "xyz.com" vs "xyz com".
    "name_trigram_jaccard",
    # Distinguishes "address missing on one/both sides" from "address present but different" -
    # the model previously saw both cases as low addr_ratio/addr_token_sort with no way to tell them apart.
    "address_missing",
]


def build_entity_store(
    source1_path: str | Path,
    source2_path: str | Path,
    source3_path: str | Path,
    database_path: str | Path,
    rebuild: bool = False,
) -> Path:
    """Materialise indexed entity tables once for fast per-chunk lookups."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists() and not rebuild:
        return database_path
    con = duckdb.connect(str(database_path))
    try:
        con.execute("DROP TABLE IF EXISTS source1")
        con.execute("DROP TABLE IF EXISTS candidates")
        columns = "entity_id, core_name, clean_address, country, legal_form, numbers"
        con.execute(f"CREATE TABLE source1 AS SELECT {columns} FROM read_parquet(?)", [str(source1_path)])
        con.execute(
            f"CREATE TABLE candidates AS SELECT {columns} FROM read_parquet(?) "
            f"UNION ALL SELECT {columns} FROM read_parquet(?)",
            [str(source2_path), str(source3_path)],
        )
        con.execute("CREATE INDEX source1_entity_id_idx ON source1(entity_id)")
        con.execute("CREATE INDEX candidates_entity_id_idx ON candidates(entity_id)")
    finally:
        con.close()
    return database_path


def _normalize_num(token: str) -> str:
    """Strip leading zeros / stray punctuation so "0345", "#345" and "345" all
    compare equal. Falls back to the raw token if it isn't purely numeric."""
    digits = "".join(ch for ch in str(token) if ch.isdigit())
    if not digits:
        return str(token)
    try:
        return str(int(digits))
    except ValueError:
        return str(token)


def _jaccard(left, right) -> float:
    left_set = {_normalize_num(x) for x in (left or [])}
    right_set = {_normalize_num(x) for x in (right or [])}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _overlap_coefficient(left_text: str, right_text: str) -> float:
    """Containment score on whitespace tokens: |A∩B| / min(|A|,|B|).

    Plain Jaccard punishes a short, correct name/address for being short:
    "ABC Technologies" vs "ABC Technologies Pvt Ltd Mumbai" scores badly on
    Jaccard but should score near 1.0 here, since every token on the shorter
    side is present in the longer one.
    """
    left_set, right_set = set(left_text.split()), set(right_text.split())
    if not left_set or not right_set:
        return 0.0
    smaller = min(len(left_set), len(right_set))
    return len(left_set & right_set) / smaller if smaller else 0.0


def _trigrams(text: str) -> set[str]:
    """Character trigrams over the space-stripped string, so a fused/split
    spelling ("shreebalaji" vs "shree balaji") still lines up character-for-
    character instead of being split into different tokens first."""
    stripped = (text or "").replace(" ", "")
    return {stripped[i:i + 3] for i in range(len(stripped) - 2)}


def _trigram_jaccard(left_text: str, right_text: str) -> float:
    left_grams, right_grams = _trigrams(left_text), _trigrams(right_text)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _exact_embedding_cosine(row: dict, embeddings: dict[str, np.ndarray] | None) -> float:
    """Compute cosine from the two persisted vectors, not the FAISS score."""
    if embeddings is None:
        score = row["retrieval_score"]
        if score is None:
            raise ValueError("embedding_paths are required when phonetic candidates are enabled")
        return float(score)
    left = embeddings["source1"][row["s1_embedding_row"]].astype(np.float32, copy=False)
    right = embeddings[row["candidate_source"]][row["candidate_embedding_row"]].astype(np.float32, copy=False)
    # Vectors were L2-normalised before float16 storage. Re-normalising guards
    # against the small norm drift introduced by float16 quantisation.
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _feature_frame(joined: pl.DataFrame, embeddings: dict[str, np.ndarray] | None = None) -> pl.DataFrame:
    """Compute expensive string measures only for one bounded candidate batch."""
    rows = joined.select(
        "s1_id", "candidate_id", "label", "retrieval_score", "candidate_source",
        "s1_embedding_row", "candidate_embedding_row",
        "left_name", "right_name", "left_address", "right_address",
        "left_country", "right_country", "left_legal", "right_legal",
        "left_numbers", "right_numbers",
    ).iter_rows(named=True)
    values = {name: [] for name in FEATURE_COLUMNS}
    identifiers = {"s1_id": [], "candidate_id": [], "label": []}
    for row in rows:
        left_name, right_name = row["left_name"] or "", row["right_name"] or ""
        left_address, right_address = row["left_address"] or "", row["right_address"] or ""
        identifiers["s1_id"].append(row["s1_id"])
        identifiers["candidate_id"].append(row["candidate_id"])
        identifiers["label"].append(row["label"])
        values["name_ratio"].append(fuzz.ratio(left_name, right_name))
        values["name_token_sort"].append(fuzz.token_sort_ratio(left_name, right_name))
        values["name_token_set"].append(fuzz.token_set_ratio(left_name, right_name))
        values["name_jaro"].append(jellyfish.jaro_winkler_similarity(left_name, right_name))
        values["addr_ratio"].append(fuzz.ratio(left_address, right_address))
        values["addr_token_sort"].append(fuzz.token_sort_ratio(left_address, right_address))
        values["numeric_jaccard"].append(_jaccard(row["left_numbers"], row["right_numbers"]))
        values["country_match"].append(int(bool(row["left_country"]) and row["left_country"] == row["right_country"]))
        values["legal_match"].append(int(bool(row["left_legal"]) and row["left_legal"] == row["right_legal"]))
        values["embedding_cosine"].append(_exact_embedding_cosine(row, embeddings))
        values["name_overlap_coef"].append(_overlap_coefficient(left_name, right_name))
        values["addr_overlap_coef"].append(_overlap_coefficient(left_address, right_address))
        values["name_trigram_jaccard"].append(_trigram_jaccard(left_name, right_name))
        values["address_missing"].append(int(not row["left_address"] or not row["right_address"]))
    return pl.DataFrame({**identifiers, **values})


def build_feature_chunks(
    labeled_pairs_path: str | Path,
    entity_store_path: str | Path,
    output_dir: str | Path,
    chunk_size: int = 100_000,
    has_labels: bool = True,
    embedding_paths: dict[str, str | Path] | None = None,
) -> Path:
    """Write feature dataset parts; no all-pairs dataframe is ever retained."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("part-*.parquet")):
        raise FileExistsError(f"{output_dir} already has feature parts; use a new directory or remove them deliberately.")
    con = duckdb.connect(str(entity_store_path), read_only=True)
    embeddings = (
        {name: np.load(path, mmap_mode="r") for name, path in embedding_paths.items()}
        if embedding_paths is not None else None
    )
    try:
        reader = pq.ParquetFile(labeled_pairs_path)
        for part, batch in enumerate(reader.iter_batches(batch_size=chunk_size)):
            print(f"Feature chunk {part + 1}: {batch.num_rows} candidate pairs", flush=True)
            con.register("pair_batch", batch)
            label_column = "p.label" if has_labels else "CAST(0 AS TINYINT)"
            joined = pl.from_arrow(con.execute("""
                SELECT p.s1_id, p.candidate_id, %s AS label, p.retrieval_score,
                       p.candidate_source, p.s1_embedding_row, p.candidate_embedding_row,
                       s.core_name AS left_name, c.core_name AS right_name,
                       s.clean_address AS left_address, c.clean_address AS right_address,
                       s.country AS left_country, c.country AS right_country,
                       s.legal_form AS left_legal, c.legal_form AS right_legal,
                       s.numbers AS left_numbers, c.numbers AS right_numbers
                FROM pair_batch p
                INNER JOIN source1 s ON p.s1_id = s.entity_id
                INNER JOIN candidates c ON p.candidate_id = c.entity_id
            """ % label_column).to_arrow_table())
            con.unregister("pair_batch")
            _feature_frame(joined, embeddings).write_parquet(output_dir / f"part-{part:06d}.parquet", compression="zstd")
            print(f"Wrote feature chunk {part + 1}.", flush=True)
    finally:
        con.close()
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    store = subparsers.add_parser("build-store")
    store.add_argument("source1")
    store.add_argument("source2")
    store.add_argument("source3")
    store.add_argument("database")
    store.add_argument("--rebuild", action="store_true")
    features = subparsers.add_parser("build-features")
    features.add_argument("labeled_pairs")
    features.add_argument("database")
    features.add_argument("output_dir")
    features.add_argument("--chunk-size", type=int, default=100_000)
    features.add_argument("--unlabeled", action="store_true", help="Candidate file has no training label column")
    args = parser.parse_args()
    if args.command == "build-store":
        build_entity_store(args.source1, args.source2, args.source3, args.database, args.rebuild)
    else:
        build_feature_chunks(args.labeled_pairs, args.database, args.output_dir, args.chunk_size, not args.unlabeled)


if __name__ == "__main__":
    main()
