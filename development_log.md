# RiskLens — Development Log

A running record of **what** was built, **why**, **how**, and **what to say about it in an interview**.
Written stage by stage as the project is built. This is the study document.

**Project:** RiskLens — Intelligent Financial Risk & Fraud Detection Platform
**Dataset:** IEEE-CIS Fraud Detection (public Kaggle benchmark, 2019, Vesta Corporation). *Not proprietary data.*
**Target:** `isFraud` (binary), ~3.5% positive rate
**Aligned to:** Citi India AIM — Data Science / AI-ML job description

---

## Plan: 10 stages

| #  | Stage                                  | Status  | JD coverage                              |
| -- | -------------------------------------- | ------- | ---------------------------------------- |
| 0  | Project setup & reproducibility        | ✅ Done | Git, venv, packaging                     |
| 1  | Data ingestion                         | ✅ Done | Python, pandas, data engineering         |
| 2  | EDA + data quality + statistics        | ⬜      | EDA, stats, SQL, visualisation           |
| 3  | Features + leakage-safe temporal split | ⬜      | Feature engineering, preprocessing       |
| 4  | Supervised modelling                   | ⬜      | Classification, imbalance handling       |
| 5  | Evaluation + calibration + risk engine | ⬜      | ROC/AUC, F1, CV, A/B testing, regression |
| 6  | Unsupervised + deep learning           | ⬜      | Clustering, anomaly detection, PyTorch   |
| 7  | Explainability                         | ⬜      | SHAP                                     |
| 8  | NLP + semantic search + RAG            | ⬜      | NLP, semantic search, RAG                |
| 9  | Agentic investigation copilot          | ⬜      | Agentic AI, prompt engineering           |
| 10 | Scale + ship                           | ⬜      | PySpark, FastAPI, Streamlit, Docker      |

**Time-box:** built in one day as a working vertical slice. Depth and study happen afterwards using this log.

### Deliberate cuts (state these honestly, don't hide them)

| Cut                                    | Why                                              | What I'd do with more time                     |
| -------------------------------------- | ------------------------------------------------ | ---------------------------------------------- |
| Temporal subsample for model iteration | Full 590k × 434 fits are minutes per run        | Fit final model on full data                   |
| No hyperparameter search               | Hours of compute for a few points of PR-AUC      | Optuna / randomised search with time-series CV |
| PySpark = demonstration script         | Data fits in memory; Spark isn't needed here     | Full port with partitioned Parquet             |
| Small local LLM (`llama3.2:3b`)      | Zero cost, runs offline, RAM-constrained machine | A frontier model for better tool-calling       |

---

# Stage 0 — Project setup & reproducibility

## What I implemented

A `src/`-layout Python package with pinned dependencies, config-driven paths, and git hygiene.

| File                       | Purpose                                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`         | Declares the`risklens` package; enables `pip install -e .` so `import risklens` works from scripts, notebooks, tests and Docker identically |
| `requirements.txt`       | Pinned versions, grouped by stage                                                                                                                 |
| `.gitignore`             | Excludes data, secrets, venv, build artefacts                                                                                                     |
| `src/risklens/config.py` | Finds project root; loads`configs/data.yaml` into a typed object                                                                                |
| `configs/data.yaml`      | Single source of truth for paths, schema and the data contract                                                                                    |

## Why

**Reproducibility is a requirement, not a nicety.** In a bank, "which data and which code produced this model?" is an audit question. Three specific problems this solves:

1. **`src/` layout + editable install** — the alternative is `sys.path` hacks or `../../` relative imports that break the moment a file moves. With `pip install -e .`, `import risklens` resolves the same way everywhere.
2. **Root discovery via `pyproject.toml`** — `config.py` walks *up* the directory tree until it finds `pyproject.toml`. No `os.chdir`, no hard-coded absolute paths. The code is location-independent.
3. **Config file, not constants in code** — every later stage needs to agree on where data lives and what the key/target/time columns are called. One YAML, one loader, no drift.

## How it works

```python
find_project_root()   # walks up from __file__ until it sees pyproject.toml
load_data_config()    # reads configs/data.yaml, returns a frozen dataclass
                      # lru_cached, so the YAML is parsed once per process
```

`DataConfig` exposes resolved **absolute** paths as properties (`cfg.transaction_csv`, `cfg.joined_parquet`), so no calling code ever builds a path by hand.

## Bug found and fixed (worth telling)

The first commit silently **excluded the ingestion source code**. Cause: `.gitignore` contained an unanchored `data/` pattern, which matches *any* directory named `data` at *any* depth — including the source package `src/risklens/data/`.

**Fix:** anchor with a leading slash — `/data/` matches only the repo-root data directory.

**Lesson:** in `.gitignore`, `foo/` means "any directory named foo, anywhere"; `/foo/` means "the foo directory at the repo root." Always verify with `git ls-files` after the first commit.

## Interview talking points

- *"I used a `src/` layout with an editable install so imports resolve identically from a notebook, a test, a script, or a container — no `sys.path` manipulation."*
- *"Paths are resolved by walking up to `pyproject.toml`, so nothing depends on the working directory."*
- *"The data contract lives in YAML, not in code, so there's one place to change a path and no possibility of two stages disagreeing."*

---

# Stage 1 — Data Ingestion

## What I implemented

A pipeline that turns two raw CSVs into one validated, typed, time-ordered Parquet table, plus a provenance manifest.

```
data/raw/train_transaction.csv  ─┐
                                 ├─→  [ingest]  ─→  data/interim/transactions_joined.parquet
data/raw/train_identity.csv     ─┘                  reports/stage01_ingest_manifest.json
```

| File                              | Purpose                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `src/risklens/data/dtypes.py`   | Memory-efficient dtype planning, column normalisation, categorical conversion |
| `src/risklens/data/validate.py` | Seven data-contract assertions, each mapped to a named failure mode           |
| `src/risklens/data/ingest.py`   | The 8-step pipeline, the`IngestManifest`, and `load_joined()`             |
| `scripts/download_data.py`      | Kaggle fetch; degrades to printed instructions if no credentials              |
| `scripts/run_ingest.py`         | CLI entry point with`--dry-run`                                             |
| `tests/test_ingest.py`          | 12 tests on synthetic fixtures — run without the dataset                     |

## Purpose

Produce **one trustworthy table** every later stage can rely on, and make four specific failures impossible:

| Failure                               | Consequence if unguarded                                               |
| ------------------------------------- | ---------------------------------------------------------------------- |
| Join fan-out                          | Every downstream count, rate and metric silently wrong                 |
| Memory blow-up                        | Cannot load the data; or swaps, and every experiment takes 40 min      |
| Schema drift (`id-01` vs `id_01`) | Trains fine, crashes in the API at serving time                        |
| Premature transformation              | **Data leakage** — inflated offline metrics, production failure |

## How it works — the 8 steps

```
1. LOAD      sniff dtypes on a 20k-row sample → build dtype plan
             → full read with that plan → normalise `id-01`→`id_01`
             → low-cardinality strings → category
2. VALIDATE  per table, BEFORE the join: required columns, key uniqueness, orphan keys
3. JOIN      LEFT join identity onto transaction, validate="one_to_one"
4. PIN TYPES key→int32, target→int8, time→int32, amount→float64
5. VALIDATE  post-join: target ∈ {0,1} and complete, time non-null, fraud rate in range
6. SORT      by TransactionDT, stable mergesort
7. MANIFEST  SHA-256 hashes, shapes, class balance, coverage, timings
8. PERSIST   Parquet (snappy) + JSON manifest
```

**Why validate twice.** Before the join tells you *which table* is broken. After tells you *the join* behaved. Only checking at the end tells you something is wrong but not what.

## Key decisions and rejected alternatives

### Decision 1 — LEFT join, not INNER *(the most important one)*

Identity data covers only a minority of transactions. INNER join was rejected for three reasons:

1. **Discards ~75% of the data** — with 3.5% positives, every fraud example is precious.
2. **Changes the population** — transactions that have device fingerprints are not a random sample. You'd train on `P(Y|X, identity present)` and deploy on `P(Y|X)`. That is **selection bias**, and the model is miscalibrated on the population it actually scores.
3. **Deletes the signal** — missing identity is plausibly *informative* (MNAR). Absent or inconsistent device fingerprints are what you'd expect from evasive behaviour. INNER throws away the most suspicious rows, invisibly.

> **Principle: absence of data is data. Preserve it; let a later stage encode it.**

### Decision 2 — dtype planning at read time

```
590,540 × 394 × 8 bytes (float64) ≈ 1.86 GB
590,540 × 394 × 4 bytes (float32) ≈ 0.93 GB
```

Rejected: `read_csv()` then `.astype('float32')` — needs **both** copies in memory at once, so it peaks *higher* than doing nothing. Instead: sample → build dtype map → pass `dtype=` to `read_csv`. The float64 version is never materialised.

**float32 precision:** 24-bit significand → exact integers to 2²⁴ = 16,777,216, relative precision ≈ 1.19 × 10⁻⁷ (~7 significant digits). Fine for anonymised counts and day-deltas.

**Exempted from downcasting:** `TransactionAmt`. It's money — at ~£31,000 float32 resolves to about 0.002, and error compounds through the sums and ratios that feature engineering will apply. Saving would have been ~2.4 MB out of a gigabyte. Bad trade.

### Decision 3 — Parquet, not CSV / Feather / HDF5

| Format               | Rejected because                                                                                                        |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| CSV                  | Re-parses text every load (~60s), stores no dtypes                                                                      |
| Feather              | Fast but not intended as a durable format                                                                               |
| HDF5                 | Capable but fragile tooling                                                                                             |
| **Parquet** ✅ | Columnar, compressed, typed, and**read natively by Spark** — makes Stage 10's PySpark port a port, not a rewrite |

### Decision 4 — ingestion does no imputation, encoding, scaling, or column-dropping

All of those are **fitted** transformations — they learn a parameter across rows. Fitting before the temporal split (Stage 3) means fitting on the future.

> **The rule: ingestion may reshape data; it may not learn from data.**

Reshaping (join, type, sort, rename) is deterministic per-row and carries no cross-row information. Learning (mean, scale, category set, feature ranking) aggregates across rows and must be quarantined behind the split.

## The data contract — 7 checks, each for a named failure

| Check                           | Catches                                                            |
| ------------------------------- | ------------------------------------------------------------------ |
| `check_required_columns`      | Source schema changed (renamed/dropped column)                     |
| `check_unique_key`            | Non-unique key → the precondition for a safe join                 |
| `check_no_row_multiplication` | Join fan-out — output rows ≠ input rows                          |
| `check_target`                | Null labels, or values outside {0,1}                               |
| `check_time_column`           | Nulls or negatives in the time column → breaks the temporal split |
| `check_fraud_rate`            | Truncated download or wrong file — the class balance drifts       |
| orphan-key check                | Mismatched file pair (train identity + test transactions)          |

**Why crash instead of warn?** A pipeline that crashes is annoying. A pipeline that silently produces subtly wrong data is *dangerous* — it produces a plausible model that makes bad risk decisions.

## Data leakage risks identified at this stage

| Risk                                                                                  | Control                                                                               |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Fitting a transform on all data (`fillna(df.mean())`, `StandardScaler().fit(df)`) | Ingestion performs no fitted transforms at all                                        |
| Treating`TransactionDT` as an ordinary integer → shuffled split                    | Sorted by time here; Stage 3 splits chronologically                                   |
| Bursty fraud → near-duplicate siblings across a shuffled split                       | Temporal split, not random                                                            |
| Using Kaggle's`test_*.csv`                                                          | Unlabelled — unusable. We hold out the last time period of the labelled data instead |
| Joining a table built with future knowledge                                           | Identity is captured at transaction time — checked, not assumed                      |

### Dataset-specific trap: label lag

Per Vesta's data description, `isFraud` is defined by a **reported chargeback**, with fraud propagated forward to transactions linked by card/email/billing address, and anything unreported within **120 days** labelled legitimate. Three consequences:

1. **Label maturity** — recent transactions don't yet have trustworthy labels. Real deployment needs a maturation window.
2. **Label noise in negatives** — unreported fraud is labelled 0. This is a **positive-unlabelled** problem in binary clothing, which caps achievable precision.
3. **Label propagation circularity** — "prior transactions on this card" partially encodes the labelling mechanism itself.

> Saying this in an interview shows you understand the **business process generating the data**, not just the CSV. This is the strongest single talking point in Stage 1.

## Testing

12 tests, all passing, **16 seconds, no dataset required**.

Synthetic fixtures deliberately reproduce every real failure mode: 200 rows at exactly 3.5% fraud, 25% identity coverage, unsorted timestamps, hyphenated `id-01`, injected NaNs.

| Test                                                 | Verifies                                                   |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| `test_hyphenated_columns_are_normalised`           | `id-01` → `id_01`                                     |
| `test_dtype_plan_downcasts_features_but_not_money` | float32 for V/card; key/target/time/amount exempt          |
| `test_low_cardinality_objects_become_category`     | 2-distinct → category; 100-distinct → left as object     |
| `test_duplicate_join_key_is_rejected`              | Contract error on duplicate key                            |
| `test_row_multiplication_is_rejected`              | Contract error on fan-out                                  |
| `test_missing_label_is_rejected`                   | Null label rejected, not coerced to 0                      |
| `test_unexpected_label_value_is_rejected`          | A stray`2` rejected                                      |
| `test_fanout_join_raises`                          | Non-unique right key fails loudly                          |
| `test_ingest_end_to_end`                           | Row count, dtypes, NaN preservation, time order, artefacts |
| `test_ingest_is_deterministic`                     | Two runs → identical frames                               |
| `test_load_joined_roundtrips`                      | Parquet round-trip preserves shape and labels              |
| `test_missing_raw_file_gives_actionable_error`     | Error names the fix                                        |

**Why synthetic, not real data?** Speed (16s vs a 1.3 GB download), CI (must run without Kaggle credentials), and precision of intent (you can't reliably *provoke* a duplicated key or a fan-out with real data). The real data is exercised once by `run_ingest.py`, which re-applies the identical checks.

## Results — REAL RUN, full dataset

Executed `scripts/run_ingest.py` on the real 678 MB dataset. Every contract
check passed and every design prediction was confirmed.

| Metric | Result | Interpretation |
| --- | --- | --- |
| transaction | 590,540 x 394 | as published |
| identity | 144,233 x 41 | as published |
| **joined** | **590,540 x 434** | 394 + 41 - 1 shared key. **Row count unchanged - no fan-out** |
| **fraud** | **20,663 (3.499%)** | matches the published ~3.5%; contract check passed |
| **imbalance** | **1 fraud per 27.6 legitimate** | the number that justifies PR-AUC over accuracy |
| **identity coverage** | **24.42%** | an INNER join would have destroyed 75.6% of the data |
| time span | 86,400 - 15,811,131 sec = **182.0 days** | the budget for the temporal split |
| **memory** | **927.8 MB** | vs ~1.9 GB as float64 - dtype planning halved it |
| **disk** | 678 MB CSV -> **76.7 MB** Parquet | **8.8x smaller** (columnar + snappy) |
| **reload** | 25.5s CSV -> **0.63s** Parquet | **40x faster** for every later stage |
| timings | load 25.5s, join 0.41s, write 5.5s | the join is essentially free once types are right |

### Final dtype spread — proof the money exemption worked

```
float32   399    anonymised V/C/D features + identity numerics
category   32    low-cardinality strings (ProductCD, card4, DeviceType, ...)
int32       2    TransactionID, TransactionDT
int8        1    isFraud
float64     1    TransactionAmt   <-- deliberately NOT downcast
```

That single `float64` is the whole argument: everything else was halved for
memory, money was not.

### Numbers to quote in an interview

- *"590,540 transactions, 434 features after the join, 3.5% fraud - about 1 in 28."*
- *"24% identity coverage, which is exactly why I used a LEFT join. An INNER would have thrown away three quarters of the data and biased the population."*
- *"Dtype planning took the in-memory footprint from ~1.9 GB to 928 MB, and Parquet took it from 678 MB on disk to 77 MB with a 40x faster reload."*
- *"Zero rows were gained or lost in the join - the row-count assertion confirms it rather than me hoping."*

## How to run

```powershell
.venv\Scripts\python.exe -m pytest                          # tests, no data needed
.venv\Scripts\python.exe scripts\download_data.py           # fetch dataset
.venv\Scripts\python.exe scripts\run_ingest.py --dry-run    # validate only
.venv\Scripts\python.exe scripts\run_ingest.py              # write artefacts
```

## How to interpret the output

- `joined` rows must equal `transaction` rows **exactly** — if not, it would have crashed
- `joined` cols = 394 + 41 − 1 (shared join key)
- `identity cov` well below 100% is **correct** — the LEFT join preserving unmatched rows
- `time span` ≈ 180 days — the budget Stage 3 spends on the temporal split
- `in memory` ≈ 0.9–1.1 GB confirms dtype planning worked; ~2 GB means it didn't

## Interview talking points

- *"Ingestion is a contract, not a file read. The value is in the assertions around `read_csv`, and each one maps to a specific failure I'd otherwise only discover at model evaluation."*
- *"A join is a claim about cardinality. 'Left join on TransactionID' really claims that TransactionID uniquely identifies a row in both tables — so I verify the claim before relying on it, and verify the row count after."*
- *"I use a LEFT join because an INNER would discard most of the data and change the population I'm modelling. And I'd argue the absence of identity data is itself a fraud signal."*
- *"I downcast features to float32 to halve memory, but exempt the monetary column — float32 gives about seven significant digits and I won't accumulate rounding error on currency to save 2 MB."*
- *"Ingestion may reshape data but may not learn from it. Anything that fits a parameter across rows goes after the temporal split."*
- *"The data is gitignored and regenerated by a script; I commit a SHA-256 manifest instead. So any model is traceable to the exact bytes that produced it."*

## Common mistakes this stage avoids

1. `read_csv()` then `.astype()` — peaks higher than not downcasting
2. INNER join because NaNs are annoying — biases the sample
3. `fillna(0)` on the target — invents non-fraud labels
4. `fillna(df.mean())` before splitting — the canonical leak
5. `train_test_split(shuffle=True)` on temporal data — wonderful, fictional AUC
6. Optimising accuracy — 96.5% for free by predicting nothing
7. `pd.concat([train, test])` before feature engineering — textbook leakage
8. Committing the dataset to git
9. No row-count check after a join — the most common silent corruption in data engineering
10. Unanchored `.gitignore` patterns — hit live in commit `e019e58`

## Git history

| Commit      | Content                                                  |
| ----------- | -------------------------------------------------------- |
| `e019e58` | Stage 1: leakage-safe data ingestion pipeline            |
| `e74532f` | Fix`.gitignore` anchoring; track `src/risklens/data` |

---

# Stage 2 — EDA + Data Quality + Statistics

⬜ **Not yet started.** Blocked on the dataset download.

**Planned:** missingness profiling, class-balance analysis, temporal drift check, distribution comparison fraud vs legitimate, hypothesis tests (chi-square for categoricals, Mann-Whitney for skewed numerics), SQL analysis via DuckDB over the Parquet, and a saved figure set.

---
