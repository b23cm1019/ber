"""Memory-bounded LightGBM training for feature parquet parts."""

from pathlib import Path

import lightgbm as lgb
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from .build_feature_chunks import FEATURE_COLUMNS


def load_training_data(feature_dir: str | Path, max_rows: int, negative_to_positive_ratio: int, seed: int) -> pl.DataFrame:
    """Hash-sample feature parts, retaining a controlled class balance in RAM."""
    path = str(Path(feature_dir) / "part-*.parquet")
    source = pl.scan_parquet(path)
    counts = source.select(pl.len().alias("all"), pl.col("label").sum().alias("positive")).collect()
    total, positives = counts.item(0, "all"), counts.item(0, "positive")
    if positives == 0:
        raise ValueError("No positive retrieved pairs. Improve retrieval recall before training.")
    positive_budget = min(positives, max(1, max_rows // (negative_to_positive_ratio + 1)))
    negative_budget = min(total - positives, positive_budget * negative_to_positive_ratio)
    # A stable hash makes sampling reproducible and avoids loading all rows first.
    hashed = pl.struct(["s1_id", "candidate_id"]).hash(seed=seed) % 1_000_000
    positive_cutoff = int(1_000_000 * positive_budget / positives)
    negative_cutoff = int(1_000_000 * negative_budget / max(1, total - positives))
    return (
        source.filter(
            ((pl.col("label") == 1) & (hashed < positive_cutoff))
            | ((pl.col("label") == 0) & (hashed < negative_cutoff))
        )
        .select(["s1_id", "label", *FEATURE_COLUMNS])
        .collect(engine="streaming")
    )


def train_model(frame: pl.DataFrame, model_config: dict, test_size: float = 0.2, random_state: int = 42):
    """Train without leaking an entity's candidates between train and validation."""
    if frame["label"].n_unique() < 2:
        raise ValueError("Training sample contains a single class")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(frame, frame["label"], groups=frame["s1_id"]))
    train, validation = frame[train_idx], frame[val_idx]
    X_train, y_train = train.select(FEATURE_COLUMNS).to_pandas(), train["label"].to_pandas()
    X_val, y_val = validation.select(FEATURE_COLUMNS).to_pandas(), validation["label"].to_pandas()
    scale_pos_weight = max(1.0, (len(y_train) - y_train.sum()) / max(1, y_train.sum()))
    model = lgb.LGBMClassifier(
        num_leaves=model_config.get("num_leaves", 64),
        learning_rate=model_config.get("learning_rate", 0.05),
        n_estimators=model_config.get("n_estimators", 500),
        n_jobs=model_config.get("n_jobs", -1),
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(model_config.get("early_stopping_rounds", 50), verbose=False)],
    )
    return model, y_val, model.predict_proba(X_val)[:, 1]
