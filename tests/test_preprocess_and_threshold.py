import unittest

import polars as pl

from src.preprocessing.preprocess_pipeline import preprocess
from src.training.optimize_threshold import optimize_threshold


class PreprocessAndThresholdTests(unittest.TestCase):
    def test_vectorised_preprocess_derives_retrieval_columns(self):
        frame = pl.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "business_name": ["Acme, Private Limited", None],
            "business_address": ["12-B Near Market", None],
            "country": ["in", None],
        })
        result = preprocess(frame)
        self.assertEqual(result["core_name"].to_list(), ["acme", ""])
        self.assertEqual(result["legal_form"].to_list(), ["private limited", ""])
        self.assertEqual(result["numbers"].to_list()[0], ["12"])
        self.assertEqual(result["retrieval_text"].to_list()[0], "acme 12 b near market IN")

    def test_threshold_uses_f05(self):
        result = optimize_threshold([1, 1, 0], [0.90, 0.60, 0.55], start=0.5, stop=0.9, step=0.1)
        self.assertIn("threshold", result)
        self.assertGreater(result["validation_f05"], 0)


if __name__ == "__main__":
    unittest.main()
