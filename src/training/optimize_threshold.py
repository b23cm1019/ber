"""Select the classification cutoff that maximises validation F0.5."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import fbeta_score


def optimize_threshold(y_true, probabilities, start: float = 0.50, stop: float = 0.95, step: float = 0.05) -> dict:
    thresholds = np.arange(start, stop + step / 2, step)
    scored = [(float(t), float(fbeta_score(y_true, probabilities >= t, beta=0.5, zero_division=0))) for t in thresholds]
    threshold, score = max(scored, key=lambda item: item[1])
    return {"threshold": threshold, "validation_f05": score, "all_scores": dict(scored)}


def save_threshold(result: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_npz", help="NPZ containing y_true and probabilities arrays")
    parser.add_argument("output_json")
    args = parser.parse_args()
    data = np.load(args.validation_npz)
    save_threshold(optimize_threshold(data["y_true"], data["probabilities"]), args.output_json)


if __name__ == "__main__":
    main()
