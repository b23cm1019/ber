"""Small, streaming-friendly I/O helpers used by pipeline stages."""

from pathlib import Path

import polars as pl


def scan_tsv(path: str | Path) -> pl.LazyFrame:
    """Return a lazy TSV scan; callers should sink results instead of collecting."""
    return pl.scan_csv(path, separator="\t", infer_schema_length=10_000)


def scan_parquet(path: str | Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
