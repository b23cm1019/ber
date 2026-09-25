import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "train"
OUT_DIR = ROOT / "data" / "sample"

OUT_DIR.mkdir(parents=True, exist_ok=True)


SOURCE1_PATH = DATA_DIR / "train_source1.tsv"
SOURCE2_PATH = DATA_DIR / "train_source2.tsv"
SOURCE3_PATH = DATA_DIR / "train_source3.tsv"
GT_PATH = DATA_DIR / "train_ground_truth.tsv"


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

GT_SAMPLE_SIZE = 100000

EXTRA_SOURCE2_NEGATIVES = 250000
EXTRA_SOURCE3_NEGATIVES = 250000

RANDOM_STATE = 42


# --------------------------------------------------
# LOAD GT
# --------------------------------------------------

print("Loading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
)

print(f"GT rows: {len(gt):,}")


# --------------------------------------------------
# SAMPLE GT ROWS
# --------------------------------------------------

sample_gt = gt.sample(
    n=min(GT_SAMPLE_SIZE, len(gt)),
    random_state=RANDOM_STATE
)

print(f"Sampled GT rows: {len(sample_gt):,}")


# --------------------------------------------------
# COLLECT ENTITY IDS
# --------------------------------------------------

source1_ids = set(sample_gt["source1_entity_id"])

source2_ids = set()
source3_ids = set()

print("Collecting positive linked entities...")

for matched in sample_gt["matched_entity_ids"]:

    if pd.isna(matched):
        continue

    for eid in matched.split(","):

        eid = eid.strip()

        if eid.startswith("S2-"):
            source2_ids.add(eid)

        elif eid.startswith("S3-"):
            source3_ids.add(eid)

print(f"Positive S2 ids: {len(source2_ids):,}")
print(f"Positive S3 ids: {len(source3_ids):,}")


# --------------------------------------------------
# LOAD SOURCES
# --------------------------------------------------

print("Loading source files...")

source1 = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype=str
)

source2 = pd.read_csv(
    SOURCE2_PATH,
    sep="\t",
    dtype=str
)

source3 = pd.read_csv(
    SOURCE3_PATH,
    sep="\t",
    dtype=str
)


# --------------------------------------------------
# ADD RANDOM NEGATIVES
# --------------------------------------------------

print("Adding random negatives...")

extra_s2 = source2[
    ~source2["entity_id"].isin(source2_ids)
].sample(
    n=min(EXTRA_SOURCE2_NEGATIVES,
          len(source2) - len(source2_ids)),
    random_state=RANDOM_STATE
)

extra_s3 = source3[
    ~source3["entity_id"].isin(source3_ids)
].sample(
    n=min(EXTRA_SOURCE3_NEGATIVES,
          len(source3) - len(source3_ids)),
    random_state=RANDOM_STATE
)

source2_ids.update(extra_s2["entity_id"])
source3_ids.update(extra_s3["entity_id"])


# --------------------------------------------------
# FILTER SOURCES
# --------------------------------------------------

print("Filtering sources...")

source1_small = source1[
    source1["entity_id"].isin(source1_ids)
]

source2_small = source2[
    source2["entity_id"].isin(source2_ids)
]

source3_small = source3[
    source3["entity_id"].isin(source3_ids)
]


# --------------------------------------------------
# SAVE
# --------------------------------------------------

print("Saving sample dataset...")

source1_small.to_csv(
    OUT_DIR / "train_source1.tsv",
    sep="\t",
    index=False
)

source2_small.to_csv(
    OUT_DIR / "train_source2.tsv",
    sep="\t",
    index=False
)

source3_small.to_csv(
    OUT_DIR / "train_source3.tsv",
    sep="\t",
    index=False
)

sample_gt.to_csv(
    OUT_DIR / "train_ground_truth.tsv",
    sep="\t",
    index=False
)

print("\nDone.")
print(f"Source1: {len(source1_small):,}")
print(f"Source2: {len(source2_small):,}")
print(f"Source3: {len(source3_small):,}")
print(f"GT     : {len(sample_gt):,}")

print(f"\nSaved to:\n{OUT_DIR}")