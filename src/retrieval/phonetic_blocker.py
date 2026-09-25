"""Country- and number-aware Double Metaphone candidate retrieval.

This secondary channel is deliberately conservative: records must share a
phonetic business-name key and first address number, with country applied when
present.  Its candidates are unioned with FAISS candidates to recover spelling
and transliteration variants that dense retrieval can miss.
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pyarrow.parquet as pq


_BLOCK_COLUMNS = ["entity_id", "phonetic_key", "first_numeric_token", "country"]


def build_phonetic_index(targets: list[tuple[str, str | Path]], database_path: str | Path, batch_size: int = 250_000) -> Path:
    """Build an on-disk lookup table whose row offsets match embedding arrays."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database_path))
    try:
        con.execute("DROP TABLE IF EXISTS phonetic_blocks")
        con.execute("""
            CREATE TABLE phonetic_blocks (
                candidate_id VARCHAR,
                candidate_source VARCHAR,
                phonetic_key VARCHAR,
                first_numeric_token VARCHAR,
                country VARCHAR,
                candidate_embedding_row BIGINT
            )
        """)
        for source_name, source_path in targets:
            offset = 0
            for batch in pq.ParquetFile(source_path).iter_batches(batch_size=batch_size, columns=_BLOCK_COLUMNS):
                frame = pl.from_arrow(batch).rename({"entity_id": "candidate_id"}).with_columns(
                    pl.lit(source_name).alias("candidate_source"),
                    pl.Series("candidate_embedding_row", np.arange(offset, offset + len(batch), dtype=np.int64)),
                )
                con.register("block_batch", frame.select(
                    "candidate_id", "candidate_source", "phonetic_key", "first_numeric_token", "country", "candidate_embedding_row"
                ).to_arrow())
                con.execute("INSERT INTO phonetic_blocks SELECT * FROM block_batch WHERE phonetic_key <> ''")
                con.unregister("block_batch")
                offset += len(batch)
        con.execute("CREATE INDEX phonetic_lookup_idx ON phonetic_blocks(phonetic_key, first_numeric_token, country)")
    finally:
        con.close()
    return database_path


def retrieve_phonetic_candidates(
    source1_path: str | Path,
    database_path: str | Path,
    output_path: str | Path,
    per_query_limit: int = 20,
    batch_size: int = 100_000,
) -> Path:
    """Retrieve bounded phonetic-block candidates for every Source1 record."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database_path), read_only=True)
    writer = None
    offset = 0
    try:
        for batch in pq.ParquetFile(source1_path).iter_batches(batch_size=batch_size, columns=_BLOCK_COLUMNS):
            queries = pl.from_arrow(batch).rename({"entity_id": "s1_id"}).with_columns(
                pl.Series("s1_embedding_row", np.arange(offset, offset + len(batch), dtype=np.int64))
            )
            con.register("query_batch", queries.select(
                "s1_id", "phonetic_key", "first_numeric_token", "country", "s1_embedding_row"
            ).to_arrow())
            table = con.execute("""
                SELECT s1_id, candidate_id, candidate_source,
                       'phonetic' AS retrieval_method,
                       CAST(NULL AS FLOAT) AS retrieval_score,
                       CAST(block_rank AS SMALLINT) AS retrieval_rank,
                       s1_embedding_row, candidate_embedding_row
                FROM (
                    SELECT q.s1_id, b.candidate_id, b.candidate_source,
                           q.s1_embedding_row, b.candidate_embedding_row,
                           row_number() OVER (
                               PARTITION BY q.s1_id
                               ORDER BY b.candidate_source, b.candidate_embedding_row
                           ) AS block_rank
                    FROM query_batch q
                    INNER JOIN phonetic_blocks b
                      ON q.phonetic_key <> ''
                     AND b.phonetic_key = q.phonetic_key
                     AND q.first_numeric_token <> ''
                     AND b.first_numeric_token = q.first_numeric_token
                     AND (q.country = '' OR b.country = q.country)
                )
                WHERE block_rank <= ?
            """, [per_query_limit]).to_arrow_table()
            con.unregister("query_batch")
            if len(table):
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
            offset += len(batch)
    finally:
        if writer is not None:
            writer.close()
        con.close()
    if writer is None:
        # Keep downstream union schema stable even when the conservative block
        # has no hits on a small development sample.
        pl.DataFrame(schema={
            "s1_id": pl.String, "candidate_id": pl.String, "candidate_source": pl.String,
            "retrieval_method": pl.String, "retrieval_score": pl.Float32, "retrieval_rank": pl.Int16,
            "s1_embedding_row": pl.Int64, "candidate_embedding_row": pl.Int64,
        }).write_parquet(output_path)
    return output_path


def union_candidate_files(candidate_paths: list[str | Path], output_path: str | Path) -> Path:
    """Union channels, retaining FAISS metadata when a pair appears in both."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        paths = [str(Path(path)) for path in candidate_paths]
        con.from_parquet(paths).create_view("all_candidates")
        con.sql("""
            SELECT * EXCLUDE (priority, duplicate_rank)
            FROM (
                SELECT *,
                       CASE WHEN retrieval_method = 'faiss' THEN 0 ELSE 1 END AS priority,
                       row_number() OVER (
                           PARTITION BY s1_id, candidate_id
                           ORDER BY CASE WHEN retrieval_method = 'faiss' THEN 0 ELSE 1 END
                       ) AS duplicate_rank
                FROM all_candidates
            )
            WHERE duplicate_rank = 1
        """).to_parquet(str(output_path), compression="zstd")
    finally:
        con.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-index")
    build.add_argument("database")
    build.add_argument("--target", action="append", nargs=2, metavar=("SOURCE", "PARQUET"), required=True)
    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("source1_parquet")
    retrieve.add_argument("database")
    retrieve.add_argument("output_path")
    retrieve.add_argument("--per-query-limit", type=int, default=20)
    union = commands.add_parser("union")
    union.add_argument("output_path")
    union.add_argument("candidate_paths", nargs="+")
    args = parser.parse_args()
    if args.command == "build-index":
        build_phonetic_index(args.target, args.database)
    elif args.command == "retrieve":
        retrieve_phonetic_candidates(args.source1_parquet, args.database, args.output_path, args.per_query_limit)
    else:
        union_candidate_files(args.candidate_paths, args.output_path)


if __name__ == "__main__":
    main()
