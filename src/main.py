"""CLI orchestrator for the scalable entity-resolution training pipeline."""

import argparse
import json

import joblib
import numpy as np
import yaml

from src.preprocessing.convert_to_parquet import convert_tsv
from src.retrieval.build_embeddings import build_embeddings
from src.retrieval.build_faiss_index import build_index
from src.retrieval.retrieve_candidates import retrieve_candidates
from src.retrieval.phonetic_blocker import build_phonetic_index, retrieve_phonetic_candidates, union_candidate_files
from src.training.generate_training_pairs import generate_training_pairs
from src.training.build_feature_chunks import build_entity_store, build_feature_chunks
from src.training.train_lgbm import load_training_data, train_model
from src.training.optimize_threshold import optimize_threshold, save_threshold
from src.utils.paths import project_path
from src.validation.check_pipeline import validate_pipeline
from src.inference.score_predictions import run_inference
from src.inference.generate_submission import generate_submission


def _paths(cfg):
    return {name: project_path(value) for name, value in cfg["paths"].items()}


def convert_all(cfg):
    paths = _paths(cfg)
    for name in ("source1", "source2", "source3"):
        convert_tsv(paths[f"train_{name}"], paths["parquet_dir"] / f"{name}.parquet")

def convert_test(cfg):
    paths = _paths(cfg)

    for name in ("source1", "source2", "source3"):
        convert_tsv(
            paths[f"test_{name}"],
            paths["parquet_dir"] / f"test_{name}.parquet"
        )

def embed_all(cfg, device=None):
    paths, retrieval = _paths(cfg), cfg["retrieval"]
    for name in ("source1", "source2", "source3"):
        build_embeddings(
            paths["parquet_dir"] / f"{name}.parquet", paths["embedding_dir"], name,
            retrieval["model_name"], retrieval["batch_size"], device,
        )

def embed_test(cfg, device=None):
    paths, retrieval = _paths(cfg), cfg["retrieval"]

    for name in ("source1", "source2", "source3"):
        build_embeddings(
            paths["parquet_dir"] / f"test_{name}.parquet",
            paths["embedding_dir"],
            f"test_{name}",
            retrieval["model_name"],
            retrieval["batch_size"],
            device,
        )


def index_targets(cfg):
    paths, retrieval = _paths(cfg), cfg["retrieval"]
    for name in ("source2", "source3"):
        build_index(
            paths["embedding_dir"] / f"{name}.npy", paths["index_dir"] / f"{name}.index",
            retrieval["hnsw_m"], retrieval["ef_construction"],
        )
    build_phonetic_index(
        [
            ("source2", paths["parquet_dir"] / "source2.parquet"),
            ("source3", paths["parquet_dir"] / "source3.parquet"),
        ],
        paths["phonetic_index"],
    )

def index_test(cfg):
    paths, retrieval = _paths(cfg), cfg["retrieval"]

    for name in ("source2", "source3"):
        build_index(
            paths["embedding_dir"] / f"test_{name}.npy",
            paths["index_dir"] / f"test_{name}.index",
            retrieval["hnsw_m"],
            retrieval["ef_construction"],
        )

    build_phonetic_index(
        [
            ("source2", paths["parquet_dir"] / "test_source2.parquet"),
            ("source3", paths["parquet_dir"] / "test_source3.parquet"),
        ],
        paths["index_dir"] / "test_phonetic_blocks.duckdb",
    )


def retrieve_train(cfg):
    paths, retrieval = _paths(cfg), cfg["retrieval"]
    retrieve_candidates(
        paths["embedding_dir"] / "source1.npy", paths["embedding_dir"] / "source1_ids.npy",
        [
            ("source2", paths["index_dir"] / "source2.index", paths["embedding_dir"] / "source2_ids.npy"),
            ("source3", paths["index_dir"] / "source3.index", paths["embedding_dir"] / "source3_ids.npy"),
        ],
        paths["faiss_candidate_pairs"], retrieval["top_k"], retrieval["query_batch_size"], retrieval["ef_search"],
    )
    retrieve_phonetic_candidates(
        paths["parquet_dir"] / "source1.parquet", paths["phonetic_index"],
        paths["phonetic_candidate_pairs"], retrieval["phonetic_per_query_limit"],
    )
    union_candidate_files(
        [paths["faiss_candidate_pairs"], paths["phonetic_candidate_pairs"]], paths["candidate_pairs"]
    )


def label_and_feature(cfg):
    paths, training = _paths(cfg), cfg["training"]
    generate_training_pairs(
        paths["candidate_pairs"], paths["train_gt"], paths["labeled_pairs"],
        training["hard_negatives_per_positive"],
    )
    build_entity_store(
        paths["parquet_dir"] / "source1.parquet", paths["parquet_dir"] / "source2.parquet",
        paths["parquet_dir"] / "source3.parquet", paths["entity_store"],
    )
    build_feature_chunks(
        paths["labeled_pairs"], paths["entity_store"], paths["feature_dir"], training["feature_chunk_size"],
        embedding_paths={
            "source1": paths["embedding_dir"] / "source1.npy",
            "source2": paths["embedding_dir"] / "source2.npy",
            "source3": paths["embedding_dir"] / "source3.npy",
        },
    )


def train(cfg):
    paths, training = _paths(cfg), cfg["training"]
    frame = load_training_data(
        paths["feature_dir"], training["max_training_rows"],
        training["negative_to_positive_ratio"], training["random_state"],
    )
    model, y_val, probabilities = train_model(frame, cfg["model"], training["test_size"], training["random_state"])
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths["model_dir"] / "model.pkl")
    result = optimize_threshold(y_val, probabilities, **cfg["threshold"])
    save_threshold(result, paths["model_dir"] / "threshold.json")
    np.savez(paths["model_dir"] / "validation_predictions.npz", y_true=np.asarray(y_val), probabilities=probabilities)
    return result

def infer(cfg):
    paths = _paths(cfg)
    retrieval = cfg["retrieval"]

    predictions = run_inference(
        query_embeddings=paths["embedding_dir"] / "test_source1.npy",
        query_ids=paths["embedding_dir"] / "test_source1_ids.npy",

        source1_parquet=paths["parquet_dir"] / "test_source1.parquet",
        source2_parquet=paths["parquet_dir"] / "test_source2.parquet",
        source3_parquet=paths["parquet_dir"] / "test_source3.parquet",

        source2_index=paths["index_dir"] / "test_source2.index",
        source2_ids=paths["embedding_dir"] / "test_source2_ids.npy",

        source3_index=paths["index_dir"] / "test_source3.index",
        source3_ids=paths["embedding_dir"] / "test_source3_ids.npy",

        model_path=paths["model_dir"] / "model.pkl",
        threshold_path=paths["model_dir"] / "threshold.json",

        work_dir=project_path("data/inference"),

        top_k=retrieval["top_k"],
        query_batch_size=retrieval["query_batch_size"],
        ef_search=retrieval["ef_search"],

        phonetic_index=paths["index_dir"] / "test_phonetic_blocks.duckdb",
        phonetic_per_query_limit=retrieval["phonetic_per_query_limit"],
    )

    submission_path = project_path(
        "submissions/submission.tsv"
    )

    generate_submission(
        paths["parquet_dir"] / "test_source1.parquet",
        predictions,
        submission_path,
    )

    print(f"\nSubmission saved to:\n{submission_path}")

def all_infer(cfg,device=None):
    convert_test(cfg)
    embed_test(cfg,device)
    index_test(cfg)
    infer(cfg)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["convert", "convert-test", "embed", "embed-test", "index", "index-test", "retrieve", "evaluate-retrieval", "label-features", "train", "validate", "infer", "all-train", "all-infer"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default=None, help="Embedding device, e.g. cuda")
    args = parser.parse_args()
    with project_path(args.config).open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if args.command == "convert":
        convert_all(cfg)
    elif args.command == "convert-test":
        convert_test(cfg)
    elif args.command == "embed":
        embed_all(cfg, args.device)
    elif args.command == "embed-test":
        embed_test(cfg,args.device)
    elif args.command == "index":
        index_targets(cfg)
    elif args.command == "index-test":
        index_test(cfg)
    elif args.command == "retrieve":
        retrieve_train(cfg)
    elif args.command == "evaluate-retrieval":
        from src.evaluation.evaluate_retrieval import evaluate_retrieval, print_retrieval_report
        paths = _paths(cfg)
        print_retrieval_report(evaluate_retrieval(paths["candidate_pairs"], paths["train_gt"]))
    elif args.command == "label-features":
        label_and_feature(cfg)
    elif args.command == "train":
        print(json.dumps(train(cfg), indent=2))
    elif args.command == "validate":
        validate_pipeline(_paths(cfg))
    elif args.command == "infer":
        infer(cfg)
    elif args.command == "all-infer":
        all_infer(cfg, args.device)
    else:
        convert_all(cfg)
        embed_all(cfg, args.device)
        index_targets(cfg)
        retrieve_train(cfg)
        from src.evaluation.evaluate_retrieval import evaluate_retrieval, print_retrieval_report
        paths = _paths(cfg)
        print_retrieval_report(evaluate_retrieval(paths["candidate_pairs"], paths["train_gt"]))
        label_and_feature(cfg)
        print(json.dumps(train(cfg), indent=2))


if __name__ == "__main__":
    main()
