"""Address-number overlap candidate retrieval.

This channel intentionally ignores the business name entirely. Its whole job
is to recover the failure mode already observed in the miss audit: records
with a *completely different* name that nevertheless share a street/house
number and country (e.g. two very different-looking names both sitting at
"345, MG Road" in the same country). Neither the FAISS text-embedding channel
nor the phonetic-name channel can retrieve this kind of pair, because both
still key off the name. This channel keys off address numbers only, so it is
unioned with the other two rather than replacing them.

We explode the ``numbers`` list column (every digit run found in the address,
e.g. house/street/PIN numbers) into one row per number, and block on exact
number-token equality plus country. Very short numeric tokens (single/double
digit, e.g. flat/floor numbers like "1" or "12") are dropped before indexing
because they are far too common and would blow up the block size without
adding precision; this mirrors the doc's advice to try "house numbers overlap
+ locality overlap" - we don't have a locality field, so country plays that
role here.
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pyarrow.parquet as pq


_BLOCK_COLUMNS = ["entity_id", "numbers", "country"]

# Numeric tokens shorter than this are dropped from the block index: they are
# common (floor/flat/unit numbers) and match too often to be useful signal.
_MIN_NUMBER_LEN = 3


def build_address_block_index(
    targets: list[tuple[str, str | Path]],
    database_path: str | Path,
    batch_size: int = 250_000,
) -> Path:
    """Build an on-disk (number_token -> candidate) inverted index."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database_path))
    try:
        con.execute("DROP TABLE IF EXISTS address_blocks")
        con.execute("""
            CREATE TABLE address_blocks (
                candidate_id VARCHAR,
                candidate_source VARCHAR,
                number_token VARCHAR,
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
                ).explode("numbers").rename({"numbers": "number_token"}).filter(
                    pl.col("number_token").is_not_null()
                    & (pl.col("number_token").str.len_chars() >= _MIN_NUMBER_LEN)
                )
                con.register("block_batch", frame.select(
                    "candidate_id", "candidate_source", "number_token", "country", "candidate_embedding_row"
                ).to_arrow())
                con.execute("INSERT INTO address_blocks SELECT * FROM block_batch")
                con.unregister("block_batch")
                offset += len(batch)
        con.execute("CREATE INDEX address_lookup_idx ON address_blocks(number_token, country)")
    finally:
        con.close()
    return database_path


def retrieve_address_candidates(
    source1_path: str | Path,
    database_path: str | Path,
    output_path: str | Path,
    per_query_limit: int = 20,
    batch_size: int = 100_000,
) -> Path:
    """Retrieve bounded address-number-overlap candidates for every Source1 record."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database_path), read_only=True)
    writer = None
    offset = 0
    try:
        for batch in pq.ParquetFile(source1_path).iter_batches(batch_size=batch_size, columns=_BLOCK_COLUMNS):
            queries = pl.from_arrow(batch).rename({"entity_id": "s1_id"}).with_columns(
                pl.Series("s1_embedding_row", np.arange(offset, offset + len(batch), dtype=np.int64))
            ).explode("numbers").rename({"numbers": "number_token"}).filter(
                pl.col("number_token").is_not_null()
                & (pl.col("number_token").str.len_chars() >= _MIN_NUMBER_LEN)
            )
            con.register("query_batch", queries.select(
                "s1_id", "number_token", "country", "s1_embedding_row"
            ).to_arrow())
            table = con.execute("""
                SELECT s1_id, candidate_id, candidate_source,
                       'address' AS retrieval_method,
                       CAST(NULL AS FLOAT) AS retrieval_score,
                       CAST(block_rank AS SMALLINT) AS retrieval_rank,
                       s1_embedding_row, candidate_embedding_row
                FROM (
                    SELECT q.s1_id, b.candidate_id, b.candidate_source,
                           q.s1_embedding_row, b.candidate_embedding_row,
                           row_number() OVER (
                               PARTITION BY q.s1_id, b.candidate_id
                               ORDER BY b.candidate_source, b.candidate_embedding_row
                           ) AS dedup_rank,
                           row_number() OVER (
                               PARTITION BY q.s1_id
                               ORDER BY b.candidate_source, b.candidate_embedding_row
                           ) AS block_rank
                    FROM query_batch q
                    INNER JOIN address_blocks b
                      ON b.number_token = q.number_token
                     AND q.country <> ''
                     AND b.country = q.country
                )
                WHERE dedup_rank = 1 AND block_rank <= ?
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
        pl.DataFrame(schema={
            "s1_id": pl.String, "candidate_id": pl.String, "candidate_source": pl.String,
            "retrieval_method": pl.String, "retrieval_score": pl.Float32, "retrieval_rank": pl.Int16,
            "s1_embedding_row": pl.Int64, "candidate_embedding_row": pl.Int64,
        }).write_parquet(output_path)
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
    args = parser.parse_args()
    if args.command == "build-index":
        build_address_block_index(args.target, args.database)
    else:
        retrieve_address_candidates(args.source1_parquet, args.database, args.output_path, args.per_query_limit)


if __name__ == "__main__":
    main()
