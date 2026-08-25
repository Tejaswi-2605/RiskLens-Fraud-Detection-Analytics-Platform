# RiskLens — Intelligent Financial Risk & Fraud Detection Platform

An end-to-end fraud detection system built on the **IEEE-CIS Fraud Detection**
dataset (public Kaggle benchmark, 2019). *Not proprietary data.*

## Problem

Given a card-not-present transaction, estimate the probability that it is
fraudulent, and turn that probability into an auditable **approve / review /
decline** decision under an explicit cost model.

Two properties make this hard, and they drive every design choice here:

1. **Severe class imbalance** — roughly 1 fraud in 29 transactions. Accuracy is
   meaningless; a model that predicts "never fraud" scores ~96.5%.
2. **Time dependence** — fraud patterns drift. Any random train/test split
   leaks the future into the past and produces a score that will not survive
   production.

## Architecture

```
IEEE-CIS Data → Ingestion → Understanding → Quality → EDA/Stats
   → Leakage-Safe Preprocessing → Feature Engineering → Temporal Split
   → Baseline ML → Advanced ML → Imbalance → Evaluation → Anomaly Detection
   → Deep Learning → Explainability → Calibration → Risk Engine
   → PySpark → FastAPI → Streamlit → Docker
```

## Stage status

| # | Stage | Status |
|---|-------|--------|
| 1 | Data Ingestion | ✅ implemented, 12 tests passing |
| 2 | Data Understanding | — |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
pip install -e .
```

## Stage 1 — Ingestion

```bash
python scripts/download_data.py     # needs ~/.kaggle/kaggle.json
python scripts/run_ingest.py        # builds the interim parquet
python -m pytest                    # 12 tests, runs without the dataset
```

Produces:

| Artefact | Committed? | Purpose |
|---|---|---|
| `data/interim/transactions_joined.parquet` | no (gitignored) | typed, joined table for all later stages |
| `reports/stage01_ingest_manifest.json` | **yes** | provenance: SHA-256 of inputs, shapes, class balance, timings |

The manifest is the reproducibility anchor: the data is regenerable from the
script, so the repo stores the *fingerprint* of the data, not the bytes.

## Layout

```
configs/data.yaml            paths, schema, and the data contract
src/risklens/config.py       root discovery + typed config
src/risklens/data/dtypes.py  memory-efficient dtype planning
src/risklens/data/validate.py data-contract assertions
src/risklens/data/ingest.py  load → validate → join → verify → persist
scripts/                     CLI entry points
tests/                       synthetic-data tests (no download required)
```

## Non-goals

Not a RAG/LLM project. No transformers, no GNN, no technology added without a
justification from the data or the problem.
