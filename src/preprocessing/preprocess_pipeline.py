"""Vectorised Polars preprocessing for entity records."""

import polars as pl

from .legal_suffix import LEGAL_SUFFIXES


# Longest first prevents ``limited`` from consuming part of ``private limited``.
_SUFFIX_PATTERN = r"(?:^|\s)(" + "|".join(
    suffix.replace(" ", r"\s+") for suffix in sorted(LEGAL_SUFFIXES, key=len, reverse=True)
) + r")$"
_LANDMARK_PATTERN = r"\b(?:near|opp|opposite|behind|beside|next)\b"


def _double_metaphone_batch(values: pl.Series) -> pl.Series:
    """Return the primary Double Metaphone code for a name batch.

    This is the one intentionally Python-backed preprocessing step. It is
    required for the secondary phonetic retrieval channel and runs per column
    batch rather than via a dataframe row loop.
    """
    try:
        from metaphone import doublemetaphone
    except ImportError as exc:
        raise RuntimeError("Install Metaphone to build phonetic retrieval blocks.") from exc
    return pl.Series(values.name, [doublemetaphone(value or "")[0] for value in values], dtype=pl.String)


def _normalise(column: str) -> pl.Expr:
    """Unicode-aware text cleanup implemented entirely by Polars expressions."""
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def preprocess(frame: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """Add normalised, retrieval, and reranking columns without Python row UDFs."""
    is_lazy = isinstance(frame, pl.LazyFrame)
    result = frame.with_columns(
        _normalise("business_name").alias("clean_name"),
        _normalise("business_address").alias("clean_address"),
        pl.col("country").cast(pl.Utf8, strict=False).fill_null("").str.to_uppercase().alias("country"),
    ).with_columns(
        pl.col("clean_name").str.extract(_SUFFIX_PATTERN, group_index=1).fill_null("").alias("legal_form"),
        pl.col("clean_name").str.replace(_SUFFIX_PATTERN, "").str.strip_chars().alias("core_name"),
        pl.col("clean_address").str.extract_all(r"\d+").alias("numbers"),
        pl.col("clean_address").str.extract_all(_LANDMARK_PATTERN).list.join(" ").alias("landmark_text"),
    ).with_columns(
        pl.col("core_name").map_batches(_double_metaphone_batch, return_dtype=pl.String).alias("phonetic_key"),
        pl.col("numbers").list.first().fill_null("").alias("first_numeric_token"),
        pl.concat_str(["core_name", "clean_address", "country"], separator=" ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("retrieval_text")
    )
    return result if is_lazy else result
