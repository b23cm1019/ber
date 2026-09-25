"""Retrieve Top-K candidates from each target FAISS index in query batches."""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq


def _load_index(path: str | Path, ef_search: int):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Install faiss-cpu (or faiss-gpu) to retrieve candidates.") from exc
    index = faiss.read_index(str(path))
    index.hnsw.efSearch = ef_search
    return index


def _load_ids(path: str | Path):
    """Load IDs lazily when they are written by build_embeddings."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return pl.read_parquet(path, columns=["entity_id"])["entity_id"].to_numpy()


def retrieve_candidates(
    query_embeddings: str | Path,
    query_ids: str | Path,
    targets: list[tuple[str, str | Path, str | Path]],
    output_path: str | Path,
    top_k: int = 20,
    query_batch_size: int = 4096,
    ef_search: int = 128,
) -> Path:
    """Write one row per (source1, retrieved target) pair.

    ``targets`` items are ``(source_label, faiss_index, aligned_ids_path)``.
    Vector row positions returned by FAISS are mapped through the aligned id file.
    """
    queries = np.load(query_embeddings, mmap_mode="r")
    source1_ids = _load_ids(query_ids)
    if len(queries) != len(source1_ids):
        raise ValueError("Query embeddings and IDs have different row counts")
    loaded = [
        (label, _load_index(index_path, ef_search), _load_ids(ids_path))
        for label, index_path, ids_path in targets
    ]
    for _, index, ids in loaded:
        if index.ntotal != len(ids):
            raise ValueError("FAISS index and aligned ids have different row counts")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for start in range(0, len(queries), query_batch_size):
            end = min(start + query_batch_size, len(queries))
            vectors = np.ascontiguousarray(queries[start:end], dtype=np.float32)
            frames = []
            for label, index, ids in loaded:
                scores, positions = index.search(vectors, top_k)
                valid = positions >= 0
                row_idx, rank = np.where(valid)
                frames.append(pl.DataFrame({
                    "s1_id": source1_ids[start:end][row_idx],
                    "candidate_id": ids[positions[valid]],
                    "candidate_source": np.full(len(row_idx), label),
                    "retrieval_method": np.full(len(row_idx), "faiss"),
                    "retrieval_score": scores[valid].astype(np.float32),
                    "retrieval_rank": (rank + 1).astype(np.int16),
                    "s1_embedding_row": (start + row_idx).astype(np.int64),
                    "candidate_embedding_row": positions[valid].astype(np.int64),
                }))
            table = pl.concat(frames).unique(["s1_id", "candidate_id"], keep="first").to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_embeddings")
    parser.add_argument("query_ids")
    parser.add_argument("output_path")
    parser.add_argument("--target", action="append", nargs=3, metavar=("LABEL", "INDEX", "IDS"), required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--ef-search", type=int, default=128)
    args = parser.parse_args()
    retrieve_candidates(args.query_embeddings, args.query_ids, args.target, args.output_path, args.top_k, args.query_batch_size, args.ef_search)


if __name__ == "__main__":
    main()
