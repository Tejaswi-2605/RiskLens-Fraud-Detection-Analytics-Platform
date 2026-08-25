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
| 10 | Serving                                | ✅ Done | FastAPI, Streamlit                       |

## Documentation index

Each stage has a **detailed teaching doc** in `docs/` (every term defined, with
tiny worked examples and interview Q&A). This log is the summary index.

| Stage | Detailed doc | Notebook |
| --- | --- | --- |
| 0-1 | [docs/stage01_ingestion.md](docs/stage01_ingestion.md) | [01_ingestion_eda_split.ipynb](notebooks/01_ingestion_eda_split.ipynb) |
| 2-3 | [docs/stage02_03_eda_and_split.md](docs/stage02_03_eda_and_split.md) | same notebook |

Notebooks are **generated and executed** by `scripts/build_notebooks.py`, so
their outputs are real results, never pasted text. Rebuild any time:

```
python scripts/build_notebooks.py
```

**Time-box:** built in one day as a working vertical slice. Depth and study happen afterwards using this log.

### Deliberate cuts (state these honestly, don't hide them)

| Cut                                    | Why                                              | What I'd do with more time                     |
| -------------------------------------- | ------------------------------------------------ | ---------------------------------------------- |
| Temporal subsample for model iteration | Full 590k × 434 fits are minutes per run        | Fit final model on full data                   |
| No hyperparameter search               | Hours of compute for a few points of PR-AUC      | Optuna / randomised search with time-series CV |
| Small local LLM (`llama3.2:3b`)      | Zero cost, runs offline, RAM-constrained machine | A frontier model for better tool-calling       |

---

# Stage 0 — Project setup & reproducibility

## What I implemented

A `src/`-layout Python package with pinned dependencies, config-driven paths, and git hygiene.

| File                       | Purpose                                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`         | Declares the`risklens` package; enables `pip install -e .` so `import risklens` works from scripts, notebooks and tests identically |
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
| **Parquet** ✅ | Columnar, compressed, typed - 8.8x smaller and 40x faster to load than CSV |

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

# Stages 2 & 3 — EDA, Statistics & the Temporal Split

> **Full teaching doc:** [docs/stage02_03_eda_and_split.md](docs/stage02_03_eda_and_split.md)

## What I implemented

| File | Purpose |
| --- | --- |
| `src/risklens/data/split.py` | Temporal split, embargo, 4 safety assertions |
| `src/risklens/eda/profile.py` | Missingness, V-blocks, temporal, missing-as-signal |
| `src/risklens/eda/stats.py` | Chi-square + Cramer's V, Mann-Whitney + Cliff's delta, PSI |
| `src/risklens/eda/plots.py` | 7 figures, each answering one decision-relevant question |
| `scripts/run_eda.py` | Split-then-explore orchestration |
| `scripts/build_notebooks.py` | Generates + executes the analysis notebook |
| `tests/test_split.py` | 14 tests |

## Why EDA comes AFTER the split

If I explore everything and then choose features from what I saw, my choices
encode knowledge of the test period. That is **human-in-the-loop leakage** and
no assertion can detect it, because it happened in my head. So the split is
fixed by a rule first, and EDA only ever sees the training partition.

## Real results — the split

```
train  438,125 rows (74.2%)  fraud 15,364 (3.507%)  127.4 days
val     73,096 rows (12.4%)  fraud  2,536 (3.469%)   26.3 days
test    73,910 rows (12.5%)  fraud  2,634 (3.564%)   26.3 days
dropped  5,409 rows (1.1%)   - embargo
```

**Why 74/12/12 and not 70/15/15:** we split by calendar time, not row count.
Volume was higher in the earlier months, so the earliest 70% of the *calendar*
holds 74.2% of the *rows*. Deliberate - real deployments retrain on "the last
six months", not "the last 400,000 rows".

Fraud rate is near-identical across partitions, so the split created neither
an easy nor an impossible test set.

## Bug the tests caught

With `embargo_days: 0`, a row landing exactly on the boundary matched **both**
train and val (`t <= train_end` and `t >= val_start` where the two are equal).
Fixed by making mask lower bounds exclusive. Interval boundaries are where
off-by-one leakage lives.

## Five findings that changed the project

### 1. My Stage 1 hypothesis was WRONG (the most valuable finding)

I predicted missing identity data would indicate evasion. The data says the
opposite:

| Column | fraud when MISSING | fraud when PRESENT |
| --- | --- | --- |
| `id_04` | **2.61%** | **10.31%** |
| `DeviceType` | 2.12% | 7.58% |

**Fraud is ~4x HIGHER when identity is PRESENT.** The cause is a **confounder**:
identity is only captured for card-not-present online transactions, which are
inherently riskier. Identity presence is a proxy for *sales channel*, not for
evasiveness.

The direction was wrong; the LEFT-join decision was right, and more strongly
than I had argued.

### 2. TransactionAmt has no predictive power

Cliff's delta = 0.0014, p = 0.78. Pick a random fraud and a random legitimate
transaction and the fraud is bigger 50.07% of the time - a coin flip.
Consistent with **card testing**: small unremarkable purchases to verify a
stolen card before selling it.

### 3. Fraud is strongly non-stationary

Weekly fraud rate swings **2.07% -> 5.08%** (2.5x). Hard evidence that a random
split would be indefensible.

### 4. 339 V-columns collapse to 14 missingness patterns

Heavy redundancy. Feature selection should operate on blocks, not columns.

### 5. No drift between train and test

All PSI < 0.06 = stable. Periods are distributionally comparable.

## Why effect size, not p-value

At 438,125 rows nearly every p-value is 0. A one-penny difference becomes
"highly significant". The p-value says a difference is real; the **effect
size** says whether it is big enough to matter. Only the second ranks features.

## Best predictors found

| Feature | Test | Effect size |
| --- | --- | --- |
| `D15` | Mann-Whitney | delta = -0.240 |
| `C1` | Mann-Whitney | delta = +0.235 |
| `C13` | Mann-Whitney | delta = -0.218 |
| `id_31` (browser) | Chi-square | V = 0.185 |
| `ProductCD` | Chi-square | V = 0.163 |
| `TransactionAmt` | Mann-Whitney | delta = 0.001 (nothing) |

Nothing is strong - the best is "small but real". That is normal: fraud
detection wins by combining many weak signals, which is what boosting does.
A single overwhelming feature would suggest leakage.

## Interview talking points

- *"I split before exploring, because if I pick features after seeing the test
  period, I become the leak and no test can catch it."*
- *"Weekly fraud rate swings 2.5x, so a random split is indefensible. I can
  show the chart rather than assert it."*
- *"I use an embargo between partitions because fraud is bursty - one
  compromised card makes many near-identical transactions minutes apart, and a
  hard boundary would split that burst across train and test."*
- *"I rank by effect size, not p-value. At 438k rows everything is
  significant, including differences of one penny."*
- *"My missingness hypothesis was wrong by 4x in the opposite direction. It's a
  channel effect - identity is only captured for card-not-present transactions.
  I'd control for ProductCD before concluding anything about device
  fingerprints."*

## Artefacts

`reports/stage02_*.csv` (6 tables), `reports/stage03_split_summary.json`,
`reports/figures/*.png` (7 figures), `notebooks/01_ingestion_eda_split.ipynb`
(32 cells, executed, 6 charts + 6 tables embedded).

---

# Stage 3b — Feature Engineering

> **Full teaching doc:** [docs/stage03b_04_05_modelling.md](docs/stage03b_04_05_modelling.md)

## What I implemented and WHY

Features come in two kinds, and confusing them is how leakage happens.

| Kind | Definition | Safe before the split? |
| --- | --- | --- |
| **Deterministic** (row-wise) | Computed from ONE row alone | Yes |
| **Fitted** (cross-row) | LEARNS a parameter across many rows | No - train only |

**The test:** *could I compute this for a single transaction arriving at the
API, with no dataset available?* If yes it is deterministic and safe anywhere.

### Deterministic features added (+28 columns, 434 -> 462)

| Feature | Purpose |
| --- | --- |
| `amt_log` | Money is right-skewed; log compresses the tail so a large transaction stops dominating the linear baseline. `log1p` not `log` because `log(0)` is -inf |
| `amt_cents`, `amt_is_round` | Real prices cluster at .00/.99/.95; card testing and currency conversion produce unusual decimals |
| `hour`, `dayofweek`, `is_night` | `TransactionDT mod 24` gives a RELATIVE hour. Origin unknown but CONSTANT, so the cycle shape is real |
| `*_isna` (16 columns) | Stage 2's headline finding made explicit. Trees split on NaN anyway, but this lets the linear baseline use it and makes it visible in SHAP |
| `n_missing` | Compact summary of the 14 correlated V-blocks in one number |
| email provider/suffix/match | Reduces cardinality without losing information; a payer/recipient mismatch is a classic account-takeover indicator |

### The fitted feature: FrequencyEncoder

Replaces a category with **how often it appeared in training**. Rare is
suspicious: a card seen once among 438,125 transactions behaves differently
from one seen 5,000 times.

**Real result:** `card1` has **12,452 distinct values**. One-hot would create
12,452 mostly-empty columns. Frequency encoding turns it into **one**
informative number.

**Why it is a scikit-learn transformer, not a helper function:** counts must
come from TRAIN only. Computing them over train+test would leak a card's
future volume into its past prediction. As a transformer inside a Pipeline,
`fit` structurally cannot see val/test - the framework enforces it rather than
my memory.

### Two columns the model is FORBIDDEN to see

`TransactionID` and `TransactionDT` both increase monotonically with time. A
tree would learn `if TransactionDT > 13,000,000 then test set` - it would
learn *which period a row came from*, not fraud. That is **leakage through the
index**.

We keep the DERIVED parts (`hour`, `dayofweek`) because `mod 24` destroys the
ordering and keeps only behaviour.

**Final feature count: 470** (435 numeric, 35 categorical).

---

# Stage 4 — Supervised Modelling

## What I implemented and WHY

Two models, deliberately.

| Model | Role |
| --- | --- |
| Logistic Regression | The **baseline** - establishes the number to beat |
| XGBoost | The **candidate** |

> If XGBoost cannot clearly beat logistic regression, the honest answer is to
> ship the logistic regression. A model you can explain to a regulator has real
> value in a bank; complexity you cannot justify does not.

Having a baseline is also the only way to answer *"is 0.31 PR-AUC good?"* -
good compared to **what**?

## Real results

| Model | PR-AUC | Lift over random | ROC-AUC | Brier |
| --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | **0.3137** | 9.0x | 0.8370 | 0.13695 |
| **XGBoost** | **0.5233** | 15.1x | 0.9053 | 0.05574 |

Random baseline PR-AUC = the base rate = **0.0347**.
Reporting PR-AUC as a *lift* matters because 0.31 means nothing on its own.

## Why XGBoost fits THIS dataset

1. **It handles NaN natively** - the single biggest reason. 229 of 434 columns
   are >50% missing, and XGBoost *learns* which side of each split missing
   values belong on. Logistic regression forces you to invent a value.
2. **Native categorical support** - no one-hot explosion on 12,452 card values.
3. **Captures interactions automatically** - fraud is conjunctive ("new device
   AND unusual hour AND rare card") and a tree path IS an interaction.
4. **Scale-invariant** - no normalisation needed.

## Imbalance: class weighting, NOT SMOTE

`scale_pos_weight = 27.5` - during training one missed fraud is penalised as
heavily as 27.5 false alarms. It changes the **loss function**, not the data.

**Why SMOTE was rejected:**

- It interpolates between real frauds committed by *different people using
  different methods*, producing transactions that could not exist.
- Expensive at 438k x 470.
- **Decisive reason:** it changes the base rate from 3.5% to 50%, so predicted
  probabilities stop being calibrated - and the Stage 5 risk engine multiplies
  probability by money.

---

# Stage 5 — Evaluation, Thresholds and the Risk Engine

> **Full teaching doc:** [docs/stage03b_04_05_modelling.md](docs/stage03b_04_05_modelling.md)

## The central idea

A model outputs a **probability**. A business needs a **decision**. Converting
one to the other requires a threshold, and choosing it is a **business**
decision, not a statistical one - it depends on what a mistake costs.

## Why not accuracy, and why not ROC-AUC

- **Accuracy:** predicting "never fraud" scores 96.5% and catches zero fraud.
- **ROC-AUC:** optimistically biased under heavy imbalance - it rewards ranking
  the easy 96.5% majority correctly. Reported, not optimised.
- **PR-AUC:** only considers the positive class. Our headline metric.
- **Brier:** measures calibration, which the risk engine depends on.

## Three ways to choose a threshold

| Strategy | The business question |
| --- | --- |
| Minimum precision | "My analysts need 50% precision or they stop trusting the queue" |
| **Alert budget** | "My team can review 500 alerts a day out of 50,000" - the most realistic constraint |
| **Cost-optimal** | "Which threshold loses the least money?" |

## The cost model

| Outcome | Cost |
| --- | --- |
| False negative (fraud let through) | **the transaction amount** - we refund the customer |
| False positive (good customer declined) | **~15** - review time plus friction |
| True positive | we recover ~90% of the value |

**The key insight: the costs are asymmetric AND one of them varies per
transaction.** A 10 transaction at 60% probability is not worth blocking
(expected saving 6, false-alarm cost 15). A 5,000 transaction at 20%
probability is (expected saving 1,000). **A lower probability justifies
blocking a larger amount** - which is why a single global probability
threshold is suboptimal and why banks think in *expected loss*.

## Real results — the risk engine

| | Value |
| --- | --- |
| Do nothing (all fraud gets through) | 396,914 |
| RiskLens at the cost-optimal threshold | 221,422 |
| **Net saving** | **175,491** (44.2% of fraud loss avoided) |
| Chosen threshold | 0.4548 |
| Precision / Recall | 27.4% / 69.8% |
| Alert rate | 8.84% |
| Fraud caught / missed | 1,770 / 766 |

**This is what makes RiskLens a risk system rather than a classification
exercise:** the deliverable is a number in currency, not a metric.

## Interview talking points

- *"I don't pick a threshold by maximising F1. A false negative costs the
  transaction amount; a false positive costs about 15 in review time. Those are
  asymmetric and one varies per transaction, so I sweep thresholds against an
  explicit cost model and report the saving against doing nothing."*
- *"I report PR-AUC as a lift over the base rate, because a random model scores
  0.035 and the absolute number is meaningless without that."*
- *"I used class weighting rather than SMOTE, mainly because SMOTE changes the
  base rate and destroys the calibration my risk engine depends on."*

---


# Stages 6 & 7 — Unsupervised Learning and Explainability

> **Full teaching doc:** [docs/stage06_07_unsupervised_and_shap.md](docs/stage06_07_unsupervised_and_shap.md)

## Stage 7 — SHAP: what I implemented and WHY

**SHAP** applies the Shapley value from cooperative game theory: if features
"cooperate" to produce a prediction, how much did each contribute? The property
that makes it trustworthy is **additivity**:

    base_value + sum(shap_values) = the actual prediction

The explanation always sums exactly to the model output, so it cannot omit a
contributing factor.

**Why explainability is not optional:** regulatory (a declined customer can
demand to know why), operational (an analyst needs to know what to investigate),
and debugging (the fastest way to find leakage).

**TreeExplainer, not KernelExplainer:** exact Shapley values need every subset
of features - 2^470 here. TreeExplainer exploits the tree structure to compute
the exact answer in polynomial time.

**Global importance = mean ABSOLUTE SHAP.** A feature pushing risk up for some
rows and down for others is important, but signed values would cancel to zero
and hide it.

**Why not XGBoost's built-in importance:** gain is computed on training data and
is biased toward high-cardinality features (`card1` has 12,452 levels, `card6`
has 2, so `card1` gets thousands more chances to split). SHAP has no such bias.

### Real results — top features by mean |SHAP|

| Feature | mean abs SHAP |
| --- | --- |
| `C13` | 0.2365 |
| `TransactionAmt` | 0.2196 |
| `C14` | 0.2041 |
| `V70` | 0.1929 |
| `card6` | 0.1906 |
| `P_emaildomain` | 0.1804 |
| `C1` | 0.1544 |
| `card1_freq` | 0.1503 |
| `C11` | 0.1271 |
| `V258` | 0.1266 |


### The leakage audit

If ONE feature holds more than 35% of total SHAP magnitude, treat it as a
leakage suspect. Stage 2 established real signal here is diffuse - nothing had
an effect size above 0.24 - so a single dominant feature usually encodes the
answer, the split, or the time period.

**Verdict:** healthy - importance is spread across many features, which is what genuine fraud signal looks like
Top feature `C13` holds
8.4% of total SHAP magnitude.

## Stage 6 — Unsupervised: what I implemented and WHY

The supervised model can only recognise fraud that RESEMBLES its training data.
That is a real weakness in an adversarial domain where criminals change methods
specifically to avoid known patterns.

### Anomaly detection (Isolation Forest)

Builds random trees and measures how many random splits it takes to isolate
each row. Outliers sit in sparse regions and isolate quickly, so **short path
length = anomalous**.

Chosen over one-class SVM (roughly quadratic - would not finish on 438k) and
Local Outlier Factor (needs a distance metric, suffers the curse of
dimensionality across 470 features). Isolation Forest is linear time, needs no
scaling, and handles mixed feature scales.

**It never sees a label.** That is the point - it must be able to flag a fraud
type nobody has labelled yet.

**Real result:** PR-AUC 0.0784,
ROC-AUC 0.6714,
**2.26x random**.

*How to read this:* it is far worse than the supervised model, and that is
expected - the supervised model had 15,364 labelled examples and this had zero.
The right benchmark is random (PR-AUC = base rate = 0.035). Beating it means
fraud genuinely is anomalous in feature space, which justifies keeping the
detector as a safety net for novel attacks.

### Fraud typologies (K-Means)

We cluster **only confirmed fraud**, not the whole dataset. Clustering
everything would rediscover the majority class; we already know which rows are
fraud and are asking a different question: **what KINDS of fraud are there?**

**Scaling is REQUIRED here** (unlike for trees) because K-Means minimises
Euclidean distance - without it, a feature ranging to 31,000 would dominate one
ranging to 1 and the clustering would degenerate into amount buckets.

Clusters are labelled automatically by how far each differs from the fraud
average in standard deviations. **A cluster called "3" is useless to an
analyst.**

### Real results — fraud typologies

| ID | Cases | Share | Avg amount | Signature |
| --- | --- | --- | --- | --- |
| 0 | 2,128 | 83.9% | 162.85 | low D11, low D2, low D15, low D10 |
| 1 | 259 | 10.2% | 134.56 | high D13, high D6, high D14, high D15 |
| 2 | 18 | 0.7% | 320.83 | high C4, high C14, high C6, high C11 |
| 3 | 99 | 3.9% | 39.69 | high C12, high C7, high C2, high C8 |
| 4 | 32 | 1.3% | 181.45 | high C5, high C9, high C13, high C14 |


---


# Stages 8 & 9 — NLP, Semantic Search, RAG and the Copilot

> **Full teaching doc:** [docs/stage08_09_genai.md](docs/stage08_09_genai.md)

## The honesty problem, and how we solved it

The IEEE-CIS dataset contains **no text**. Rather than invent a fake "customer
comment" column, we generate the text a real fraud operation actually produces:
the **case narrative** an analyst writes when investigating an alert.

    structured transaction + SHAP reason codes -> case narrative -> embeddings
        -> semantic search -> "have we seen this before?"

**The architectural point worth defending:** the GenAI layer **consumes** the
ML system rather than replacing it. The model predicts, SHAP explains, the LLM
reads and summarises. Neither does the other's job.

## Stage 8a — narratives via TEMPLATE, not LLM

| Reason | Explanation |
| --- | --- |
| Faithfulness | A template **cannot hallucinate** a driver SHAP did not produce |
| Reproducibility | The same alert always yields the same text |
| Cost | Free and instant, so we can build a whole corpus |

The LLM's job is to **reason over** these facts, never to invent them. Keeping
generation separate from reasoning is what keeps the system auditable.

## Stage 8b — semantic search

Keyword search fails the analyst's actual question. A query for *"night-time
purchase from a new browser"* shares **zero words** with a case written as
*"overnight transaction on an unrecognised device"*, yet they mean the same
thing.

`all-MiniLM-L6-v2`, 384 dimensions, ~90 MB, runs **locally** so no transaction
data leaves the machine.

> This is the one place a transformer is justified in RiskLens. The brief said
> no BERT without a clear NLP requirement - semantic search genuinely requires
> learned sentence representations. We are NOT using a transformer to classify
> fraud; XGBoost does that.

**FAISS `IndexFlatIP`** - exhaustive exact search. Approximate indexes (IVF,
HNSW) trade accuracy for speed and only pay off in the millions of vectors. At
our size exact search is already milliseconds. **Being able to explain why you
did NOT reach for the fancier option is worth more than having used it.**

Vectors are L2-normalised, which makes the inner product *equal* cosine
similarity - we want the angle between meanings, not their magnitude.

**Search demo:** `"overnight transaction on an unrecognised device with a rare card"`

## Stage 8c — RAG over fraud policy

Ask an LLM about your policy and it will produce a confident, plausible,
**invented** answer. In a compliance context that is an incident, not a bug.

RAG changes the question: **retrieve** relevant policy passages, **augment** the
prompt with them, **generate** an answer grounded in them. The model stops being
a knowledge source and becomes a reading-comprehension engine over documents we
control.

The system prompt's most important rule is the **explicit refusal phrase**.
Without a permitted way to say "I don't know", a model told to answer *will*
answer - from general knowledge if it must.

Temperature **0.1**: for policy guidance, the same question must produce the
same answer.

### Real RAG results

| Question | Sources cited | Grounded | Verdict |
| --- | --- | --- | --- |
| What action is required for a HIGH risk band transaction a... | 01_risk_scoring_and_decisions.md, 04_acc | 0.824 | well grounded |
| How should decision thresholds be set, and why is maximisi... | 01_risk_scoring_and_decisions.md | 0.789 | well grounded |
| What is the label maturity window and why does it matter f... | 03_chargebacks_and_labels.md, 06_model_g | 0.919 | well grounded |
| What are the primary indicators of account takeover?... | 04_account_takeover.md, README.md | 0.982 | well grounded |
| What is the capital of France?... | 03_chargebacks_and_labels.md, 06_model_g | 1.0 | correct refusal - the model de |


The final question is deliberately out of scope. **A correctly configured RAG
system should refuse it** - that refusal is the safety property we want. A
system that happily answers off-corpus questions will also happily invent policy.

## Stage 9 — the investigation copilot

Five tools, and the tools **are** the rest of RiskLens: `score_transaction`
(Stage 4), `explain_alert` (Stage 7), `find_similar_cases` (Stage 8a/b),
`lookup_policy` (Stage 8c), `get_transaction`.

**Two modes, and we default to the LESS impressive one:**

| Mode | Description |
| --- | --- |
| `agent_loop()` | True tool-calling - the LLM decides what to call |
| **`investigate()`** | **Default.** Fixed five-step workflow, LLM writes the summary |

**Why default to the deterministic workflow:** a 3B local model calls tools
unreliably - it invents arguments and skips steps. In a regulated setting, an
investigation that *sometimes* omits the policy check is worse than one that
always runs the same five. A fixed order also means two cases are comparable.

The agent loop has a **turn cap**. That cap is the difference between an agent
and a runaway process.

**Real investigation:** tools called ->
get_transaction -> score_transaction -> explain_alert -> find_similar_cases -> required_action -> lookup_policy

---

# Stage 10 — Serving

> **Full teaching doc:** [docs/stage10_deployment.md](docs/stage10_deployment.md)

## The concept that matters most: training/serving skew

When features computed at **serving** time differ from those at **training**
time - a recomputed imputation median, a different frequency map, a different
column order, a different DTYPE - the model receives inputs it was never
trained on and degrades **silently**. No error is raised.

**Our defence is structural.** The API loads the SAME artefacts the training
run produced:

    models/xgboost.joblib            the fitted model
    models/frequency_encoder.joblib  the fitted encoder, TRAIN-time counts
    models/feature_names.joblib      the exact column list, in order
    models/feature_schema.joblib     the exact dtypes, TRAIN category lists

Nothing is re-fitted and nothing is recomputed.

## Three real bugs, all found by actually running it

1. **KeyError on the first live request.** The frequency encoder indexed every
   fitted column directly, but a real payment message is sparse and omits
   `id_31`, `DeviceInfo` and others. Absent columns now still EMIT their
   `_freq` feature - dropping the column would shift every downstream one, and
   XGBoost matches positionally.

2. **Dtype mismatch.** Training used pandas `category`; serving produced
   `object`. The raised error was the LUCKY outcome - a category is an integer
   CODE plus a list, and XGBoost splits on codes, so a serving-built list means
   "visa" could be code 0 here and code 3 at training. Silent, wrong.
   `export_feature_schema.py` now persists the training `CategoricalDtype`.

3. **Narratives printed "nan".** `_fmt` tested `isinstance(v, float)` before
   `isnan`, but numpy `float32` is not a Python `float`, so the check never
   fired. Now uses `pd.isna` and renders "not recorded".

## FastAPI

Chosen over Flask for automatic Pydantic validation and generated docs.
`TransactionAmt: float = Field(..., gt=0)` rejects a negative amount with a
clear 422 BEFORE our code runs - otherwise `log1p(-50)` produces NaN which
propagates silently into the model.

**Almost every field is optional** because real payment messages are sparse,
and Stage 2 proved which fields are absent is genuine signal.

**`/health` is deliberately honest** - it reports `degraded` when the model
failed to load. A health check returning `ok` while every request 503s is worse
than none.

## Streamlit

Built for a fraud **analyst**: risk bands not floats, plain-English reason
codes not `V257 = 3.2`, precedent, and policy with citation. Two input modes -
hand-built (relative movement) and a real validation row (full 470 features,
so genuine CRITICAL/DECLINE outcomes with the true label revealed).

`@st.cache_resource` is essential: Streamlit reruns the whole script on every
interaction.

## Deliberately NOT included

**Distributed compute.** The data is 590,540 rows and fits in 928 MB. Spark
would be slower than pandas here, and reaching for it would signal
buzzword-following rather than reasoning about fit. Knowing when NOT to use a
tool is part of the job.

---

# Final summary — what to say about RiskLens in one minute

RiskLens is an end-to-end fraud detection platform on the IEEE-CIS benchmark:
590,540 transactions, 470 engineered features, 3.5% fraud.

The thing I would emphasise is **leakage discipline**. Fraud data is temporal
and adversarial, so I split chronologically with an embargo between partitions,
I explore only the training period so my own feature choices cannot encode the
future, and every fitted transformation lives inside an sklearn Pipeline where
it structurally cannot see validation data.

I chose metrics for the problem rather than the default: PR-AUC reported as a
lift over the base rate, because accuracy is 96.5% for free. And I convert the
probability into a decision using an explicit cost model - a false negative
costs the transaction amount, a false positive costs review time - so the
deliverable is a saving in currency rather than a metric.

On top of that sits an analyst copilot: SHAP reason codes turned into case
narratives, semantic search over them for precedent, and RAG over the fraud
policy so guidance is cited rather than invented. The GenAI layer consumes the
ML system; it does not replace it.

The most instructive thing I found was that I was **wrong**. I predicted missing
identity data would indicate evasion; fraud turned out to be four times higher
when identity was *present*. It is a channel effect - identity is only captured
for card-not-present transactions, which are riskier. My join decision was
right for a reason I had not understood.
