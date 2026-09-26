"""Create normalised SentenceTransformer embeddings in bounded batches."""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import polars as pl


def _encoder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # keeps preprocessing usable without GPU packages
        raise RuntimeError("Install sentence-transformers to build embeddings.") from exc
    return SentenceTransformer(model_name)


def build_embeddings(
    source_parquet: str | Path,
    output_dir: str | Path,
    source_name: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    device: str | None = None,
) -> tuple[Path, Path]:
    """Write ``<source>.npy`` and an aligned memory-mappable ID array.

    The vector array is a NumPy memmap while encoding, so it never accumulates
    all embeddings in RAM.  The source parquet must contain entity_id and
    retrieval_text (produced by ``convert_to_parquet``).
    """
    source_parquet, output_dir = Path(source_parquet), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = pq.ParquetFile(source_parquet).metadata.num_rows
    if total_rows == 0:
        raise ValueError(f"No rows found in {source_parquet}")

    model = _encoder(model_name if device is None else model_name)
    # Probe once to obtain dimension, then re-use that result for the first batch.
    batches = pq.ParquetFile(source_parquet).iter_batches(
        batch_size=batch_size, columns=["entity_id", "retrieval_text"]
    )
    try:
        first = next(batches)
    except StopIteration as exc:
        raise ValueError(f"No rows found in {source_parquet}") from exc
    first_df = pl.from_arrow(first)
    first_vectors = model.encode(
        first_df["retrieval_text"].to_list(), batch_size=batch_size,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        device=device,
    ).astype(np.float16, copy=False)

    embedding_path = output_dir / f"{source_name}.npy"
    ids_path = output_dir / f"{source_name}_ids.npy"
    vectors = np.lib.format.open_memmap(
        embedding_path, mode="w+", dtype=np.float16, shape=(total_rows, first_vectors.shape[1])
    )
    # Entity IDs are short, fixed-width identifiers (e.g. S2-166376419). A
    # memmap avoids loading 10M target IDs while translating FAISS positions.
    ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype="<U32", shape=(total_rows,))
    offset = 0
    try:
        for batch_df, batch_vectors in _batches_with_first(first_df, first_vectors, batches, model, batch_size, device):
            end = offset + len(batch_df)
            print(f"Encoded {end:,}/{total_rows:,} rows for {source_name}", flush=True)
            vectors[offset:end] = batch_vectors
            batch_ids = batch_df["entity_id"].to_numpy()
            if any(len(value) > 32 for value in batch_ids):
                raise ValueError("entity_id exceeds the 32-character embedding ID storage limit")
            ids[offset:end] = batch_ids
            offset = end
    finally:
        vectors.flush()
        ids.flush()
    if offset != total_rows:
        raise RuntimeError(f"Embedding row count mismatch: wrote {offset}, expected {total_rows}")
    return embedding_path, ids_path


def _batches_with_first(first_df, first_vectors, remaining, model, batch_size, device):
    yield first_df, first_vectors
    for batch in remaining:
        frame = pl.from_arrow(batch)
        vectors = model.encode(
            frame["retrieval_text"].to_list(), batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
            device=device,
        ).astype(np.float16, copy=False)
        yield frame, vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_parquet")
    parser.add_argument("output_dir")
    parser.add_argument("source_name")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None, help="e.g. cuda or cpu")
    args = parser.parse_args()
    build_embeddings(args.source_parquet, args.output_dir, args.source_name, args.model, args.batch_size, args.device)


if __name__ == "__main__":
    main()
