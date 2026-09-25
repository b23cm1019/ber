from pathlib import Path


def validate_pipeline(paths):

    checks = []

    required = {
        "parquet_dir": paths["parquet_dir"],
        "embedding_dir": paths["embedding_dir"],
        "index_dir": paths["index_dir"],
        "candidate_pairs": paths["candidate_pairs"],
        "labeled_pairs": paths["labeled_pairs"],
        "feature_dir": paths["feature_dir"],
        "entity_store": paths["entity_store"],
        "model_dir": paths["model_dir"],
    }

    for name, path in required.items():

        if Path(path).exists():
            print(f"[OK] {name}")
            checks.append(True)
        else:
            print(f"[FAIL] {name}")
            checks.append(False)

    print()
    print(f"Passed {sum(checks)}/{len(checks)} checks")

    return all(checks)