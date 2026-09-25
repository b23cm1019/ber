"""Build a persisted FAISS HNSW inner-product index from normalised vectors."""

import argparse
from pathlib import Path

import numpy as np


def build_index(
    embeddings_path: str | Path,
    index_path: str | Path,
    m: int = 32,
    ef_construction: int = 200,
    add_batch_size: int = 100_000,
) -> Path:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Install faiss-cpu (or faiss-gpu) to build indexes.") from exc
    vectors = np.load(embeddings_path, mmap_mode="r")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    for start in range(0, len(vectors), add_batch_size):
        index.add(np.ascontiguousarray(vectors[start:start + add_batch_size], dtype=np.float32))
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_path")
    parser.add_argument("index_path")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    args = parser.parse_args()
    build_index(args.embeddings_path, args.index_path, args.m, args.ef_construction)


if __name__ == "__main__":
    main()
