# Business entity resolution

This repository uses a retrieval -> reranking -> classification pipeline designed
for the 12.5M-record training scale. It does not create an all-to-all candidate
matrix or retain all pairs in memory.

## Training pipeline

Run stages independently so expensive artefacts can be reused:

```powershell
python -m src.main convert
python -m src.main embed --device cuda
python -m src.main index
python -m src.main retrieve
python -m src.main evaluate-retrieval
python -m src.main label-features
python -m src.main train
```

`python -m src.main all-train --device cuda` executes the same sequence.
Adjust paths, batch sizes, HNSW parameters, and model settings in
`configs/config.yaml` before running at full scale.

Intermediate artefacts are:

```text
data/parquet/source{1,2,3}.parquet        preprocessed, columnar entities
data/embeddings/source*.npy                normalised dense vectors + aligned ID memmaps
data/indexes/source{2,3}.index             FAISS HNSW inner-product indexes
data/indexes/phonetic_blocks.duckdb         Double Metaphone + address-number blocker
data/candidates/*.parquet                  FAISS/phonetic-unioned, ranked and labelled pairs
data/features/train/part-*.parquet         100k-pair reranker feature chunks
models/lgbm/model.pkl + threshold.json     trained reranker and F0.5 cutoff
```

`evaluate-retrieval` prints pair-level Recall@20, Recall@50, and Recall@100.
Candidate files retain retrieval rank and embedding offsets, so the feature
builder computes exact cosine similarity from the saved float16 vectors rather
than reusing an approximate index score. Training keeps each positive pair and
the five highest-scoring false candidates for that Source1 record.

## Inference

For a preprocessed test Source1, create its embeddings, then use
`python -m src.inference.run_inference run --help` for the one-command
retrieval -> feature -> scoring interface. It writes `predictions.parquet` in
its work directory. Generate the submission with:

```powershell
python -m src.inference.generate_submission data/parquet/test_source1.parquet outputs/predictions.parquet outputs/submission.tsv
```

The retrieval and feature commands are reusable modules with CLIs; see their
`--help` output for test-source paths. Candidate vectors are normalised, so the
retrieval score used by the reranker is embedding cosine similarity.
