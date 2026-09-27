"""Score chunked pair features with the trained reranker and cutoff."""

import argparse
import json
from pathlib import Path

import joblib
import polars as pl
import pyarrow.parquet as pq

from src.training.build_feature_chunks import FEATURE_COLUMNS
from src.training.build_feature_chunks import build_entity_store, build_feature_chunks
from src.retrieval.retrieve_candidates import retrieve_candidates
from src.retrieval.phonetic_blocker import retrieve_phonetic_candidates, union_candidate_files
from src.retrieval.address_blocker import retrieve_address_candidates


def score_feature_chunks(feature_dir, model_path, threshold_path, output_path, batch_size=100_000):
    """Score one feature part at a time and persist candidate-level predictions."""
    model = joblib.load(model_path)
    threshold = json.loads(Path(threshold_path).read_text(encoding="utf-8"))["threshold"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for part in sorted(Path(feature_dir).glob("part-*.parquet")):
            for batch in pq.ParquetFile(part).iter_batches(batch_size=batch_size):
                frame = pl.from_arrow(batch)
                probability = model.predict_proba(frame.select(FEATURE_COLUMNS).to_pandas())[:, 1]
                result = frame.select("s1_id", "candidate_id").with_columns(
                    pl.Series("match_probability", probability),
                    pl.Series("is_match", probability >= threshold),
                )
                table = result.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return output_path


def run_inference(
    query_embeddings, query_ids, source1_parquet, source2_parquet, source3_parquet,
    source2_index, source2_ids, source3_index, source3_ids,
    model_path, threshold_path, work_dir, top_k=50, query_batch_size=4096, ef_search=128,
    phonetic_index=None, phonetic_per_query_limit=20,
    address_index=None, address_per_query_limit=20,
):
    """Run retrieval, chunked feature generation, and scoring for a test source."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    candidates = work_dir / "candidate_pairs.parquet"
    faiss_candidates = work_dir / "faiss_candidate_pairs.parquet"
    phonetic_candidates = work_dir / "phonetic_candidate_pairs.parquet"
    address_candidates = work_dir / "address_candidate_pairs.parquet"
    entity_store = work_dir / "entity_store.duckdb"
    features = work_dir / "features"
    predictions = work_dir / "predictions.parquet"
    retrieve_candidates(
        query_embeddings, query_ids,
        [("source2", source2_index, source2_ids), ("source3", source3_index, source3_ids)],
        faiss_candidates, top_k, query_batch_size, ef_search,
    )
    phonetic_index = Path(phonetic_index) if phonetic_index else Path(source2_index).parent / "phonetic_blocks.duckdb"
    retrieve_phonetic_candidates(source1_parquet, phonetic_index, phonetic_candidates, phonetic_per_query_limit)
    address_index = Path(address_index) if address_index else Path(source2_index).parent / "address_blocks.duckdb"
    retrieve_address_candidates(source1_parquet, address_index, address_candidates, address_per_query_limit)
    union_candidate_files([faiss_candidates, phonetic_candidates, address_candidates], candidates)
    build_entity_store(source1_parquet, source2_parquet, source3_parquet, entity_store, rebuild=True)
    build_feature_chunks(
        candidates, entity_store, features, has_labels=False,
        embedding_paths={
            "source1": query_embeddings,
            "source2": Path(source2_ids).with_name(Path(source2_ids).name.replace("_ids.npy", ".npy")),
            "source3": Path(source3_ids).with_name(Path(source3_ids).name.replace("_ids.npy", ".npy")),
        },
    )
    return score_feature_chunks(features, model_path, threshold_path, predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Score already-created feature chunks")
    score.add_argument("feature_dir")
    score.add_argument("model_path")
    score.add_argument("threshold_path")
    score.add_argument("output_path")
    run = commands.add_parser("run", help="Run retrieval through scoring for a test Source1")
    for name in ("query_embeddings", "query_ids", "source1_parquet", "source2_parquet", "source3_parquet",
                 "source2_index", "source2_ids", "source3_index", "source3_ids", "model_path", "threshold_path", "work_dir"):
        run.add_argument(name)
    run.add_argument("--top-k", type=int, default=50)
    run.add_argument("--query-batch-size", type=int, default=4096)
    run.add_argument("--ef-search", type=int, default=128)
    run.add_argument("--phonetic-index", default=None, help="Defaults to phonetic_blocks.duckdb beside Source2's FAISS index")
    run.add_argument("--phonetic-per-query-limit", type=int, default=20)
    run.add_argument("--address-index", default=None, help="Defaults to address_blocks.duckdb beside Source2's FAISS index")
    run.add_argument("--address-per-query-limit", type=int, default=20)
    args = parser.parse_args()
    if args.command == "score":
        score_feature_chunks(args.feature_dir, args.model_path, args.threshold_path, args.output_path)
    else:
        arguments = vars(args)
        arguments.pop("command")
        run_inference(**arguments)


if __name__ == "__main__":
    main()
