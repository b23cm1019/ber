import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.preprocessing.preprocess_pipeline import preprocess
from src.retrieval.phonetic_blocker import (
    build_phonetic_index,
    retrieve_phonetic_candidates,
    union_candidate_files,
)
from src.training.build_feature_chunks import build_entity_store, build_feature_chunks
from src.training.generate_training_pairs import generate_training_pairs


class PriorityOneTwoPipelineTests(unittest.TestCase):
    def _source(self, entity_id, business_name, business_address, country):
        return preprocess(pl.DataFrame({
            "entity_id": entity_id,
            "business_name": business_name,
            "business_address": business_address,
            "country": country,
        }))

    def test_hybrid_blocking_exact_cosine_and_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1 = self._source(["S1-1"], ["Robert Trading"], ["12 Main Street"], ["US"])
            source2 = self._source(["S2-1"], ["Rupert Trading"], ["12 Main St"], ["US"])
            source3 = self._source(["S3-1"], ["Unrelated Bakery"], ["99 Other Rd"], ["US"])
            paths = {name: root / f"{name}.parquet" for name in ("source1", "source2", "source3")}
            source1.write_parquet(paths["source1"])
            source2.write_parquet(paths["source2"])
            source3.write_parquet(paths["source3"])

            block_db = root / "blocks.duckdb"
            build_phonetic_index([("source2", paths["source2"]), ("source3", paths["source3"])], block_db)
            phonetic = root / "phonetic.parquet"
            retrieve_phonetic_candidates(paths["source1"], block_db, phonetic)
            self.assertEqual(pl.read_parquet(phonetic)["candidate_id"].to_list(), ["S2-1"])

            faiss = root / "faiss.parquet"
            pl.DataFrame({
                "s1_id": ["S1-1"], "candidate_id": ["S2-1"], "candidate_source": ["source2"],
                "retrieval_method": ["faiss"], "retrieval_score": [0.75], "retrieval_rank": [1],
                "s1_embedding_row": [0], "candidate_embedding_row": [0],
            }).write_parquet(faiss)
            candidates = root / "candidates.parquet"
            union_candidate_files([faiss, phonetic], candidates)
            combined = pl.read_parquet(candidates)
            self.assertEqual(combined.height, 1)
            self.assertEqual(combined["retrieval_method"].item(), "faiss")

            truth = root / "truth.tsv"
            pl.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1"]}).write_csv(truth, separator="\t")
            self.assertEqual(evaluate_retrieval(candidates, truth, [1]), {1: 1.0})

            labeled = combined.with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
            labeled_path = root / "labeled.parquet"
            labeled.write_parquet(labeled_path)
            store = root / "entities.duckdb"
            build_entity_store(paths["source1"], paths["source2"], paths["source3"], store)
            embeddings = {}
            for source in ("source1", "source2", "source3"):
                vector = np.array([[1.0, 0.0]], dtype=np.float16) if source != "source3" else np.array([[0.0, 1.0]], dtype=np.float16)
                vector_path = root / f"{source}.npy"
                np.save(vector_path, vector)
                embeddings[source] = vector_path
            feature_dir = root / "features"
            build_feature_chunks(labeled_path, store, feature_dir, embedding_paths=embeddings)
            self.assertAlmostEqual(pl.read_parquet(feature_dir / "part-000000.parquet")["embedding_cosine"].item(), 1.0)

    def test_hard_negative_mining_keeps_closest_false_pair_per_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = pl.DataFrame({
                "s1_id": ["S1-1", "S1-1", "S1-1"],
                "candidate_id": ["S2-positive", "S2-hard", "S2-easy"],
                "candidate_source": ["source2"] * 3,
                "retrieval_method": ["faiss"] * 3,
                "retrieval_score": [0.99, 0.98, 0.10],
                "retrieval_rank": [1, 2, 3],
                "s1_embedding_row": [0] * 3,
                "candidate_embedding_row": [0, 1, 2],
            })
            candidate_path, truth_path, output_path = root / "candidates.parquet", root / "truth.tsv", root / "labeled.parquet"
            candidates.write_parquet(candidate_path)
            pl.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-positive"]}).write_csv(truth_path, separator="\t")
            generate_training_pairs(candidate_path, truth_path, output_path, hard_negatives_per_positive=1)
            self.assertEqual(set(pl.read_parquet(output_path)["candidate_id"].to_list()), {"S2-positive", "S2-hard"})


if __name__ == "__main__":
    unittest.main()
