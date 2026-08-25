# RiskLens

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.1-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-337AB7)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.46-0088CC)](https://shap.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%20CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-1.9-4267B2)](https://faiss.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.1-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![tests](https://img.shields.io/badge/tests-62%20passing-success)](tests/)

**Fraud risk scoring and investigation platform for card payments.**

Built on the IEEE-CIS Fraud Detection dataset — a public benchmark released by
Vesta Corporation in 2019. *Not proprietary data.*

`590,540 transactions` · `504 engineered features` · `3.5% fraud` · `182 days`

---

## What it does

Given a card transaction, RiskLens estimates the probability that it is
fraudulent and converts that probability into an auditable **approve / review /
decline** decision under an explicit cost model — then explains the decision in
language a fraud analyst can act on.

Two properties of the problem drive every design choice:

**Severe class imbalance.** One fraud per 27.6 legitimate transactions.
Accuracy is meaningless here — a model that predicts "never fraud" scores
96.5%.

**Time dependence.** Fraud patterns drift; the measured weekly fraud rate
swings between 2.07% and 5.08%. Any evaluation that shuffles the data leaks the
future into the past and produces a score that will not survive production.

---

## Results

Measured on a held-out **test period** — opened once, at a decision threshold
chosen on validation and applied unchanged.

| Metric | Value |
|---|---|
| **PR-AUC** | **0.5144** (14.4× a random model) |
| ROC-AUC | 0.8969 |
| Brier score (calibrated) | 0.0238 |
| Expected Calibration Error | 0.0067 |
| Precision / Recall | 29.7% / 66.0% |
| Alert rate | 7.91% |
| **Fraud loss avoided** | **45.6%** (£180,354 over 26 days) |
| Generalisation gap (test − validation) | −0.043 |

A logistic-regression baseline scores PR-AUC **0.3137**. A random model scores
the base rate, **0.0347** — which is why PR-AUC is always reported here as a
lift rather than as a bare number.

### Operating points

The right threshold depends on how many alerts a fraud team can review:

| Alert budget | Precision | Recall |
|---|---|---|
| 0.5% of traffic | 94.0% | 13.6% |
| 1% | 88.0% | 25.4% |
| 2% | 71.1% | 41.0% |
| 5% | 42.2% | 60.8% |

---

## Architecture

```
                    data/raw/*.csv          immutable, hashed, never committed
                          │
        ┌─────────────────▼─────────────────┐
        │  1. INGESTION                      │  load → validate → join →
        │     data contract, 7 assertions    │  verify → sort → persist
        └─────────────────┬─────────────────┘
                          │  transactions_joined.parquet  (77 MB, typed)
        ┌─────────────────▼─────────────────┐
        │  2. TEMPORAL SPLIT                 │  ◀── the leakage firewall
        │     74% / 12% / 12% + 1d embargo   │
        └─────────────────┬─────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │  3. FEATURES  (434 → 504)          │
        │     • deterministic, row-wise      │  log amount, cyclical time,
        │     • causal entity aggregates     │  missingness flags, velocity,
        │     • fitted frequency encoding    │  running spend per entity
        └─────────────────┬─────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │  4. MODELS                         │  LogisticRegression baseline
        │     baseline → XGBoost             │  XGBoost, class-weighted
        └─────────────────┬─────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │  5. RISK ENGINE                    │  calibration → cost model →
        │     probability → decision         │  threshold → approve/review/decline
        └─────────────────┬─────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │  6. EXPLAINABILITY   SHAP          │  per-alert reason codes
        │  7. UNSUPERVISED     IsolationForest, KMeans typologies
        │  8. RETRIEVAL        embeddings → FAISS → semantic case search
        │  9. COPILOT          RAG over policy + tool-using agent
        └─────────────────┬─────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │  10. SERVING                       │
        │      FastAPI :8000  ·  Streamlit :8502
        └────────────────────────────────────┘
```

### The rule the whole pipeline obeys

> **Reshape data freely. Never *learn* from data before the split.**

Joining, retyping, renaming and sorting are per-row operations and carry no
information between rows. Anything that computes a statistic *across* rows — an
imputation median, a scaler, a frequency map, a calibration curve — is fitted on
the training partition only.

Entity aggregates use **expanding, backward-only windows**: for each
transaction, statistics over only that entity's *earlier* transactions. That is
causal by construction, so there is no fitting step that could contaminate
validation or test.

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| Data | **pandas**, **PyArrow / Parquet** | columnar storage: 8.8× smaller and 40× faster to load than CSV |
| Analysis | **NumPy**, **SciPy**, **matplotlib**, **DuckDB** | statistics and SQL over the Parquet without a server |
| Modelling | **scikit-learn**, **XGBoost** | XGBoost handles NaN natively — 229 of 434 raw columns are >50% missing |
| Calibration | **scikit-learn** (isotonic, Platt) | the risk engine multiplies probability by money, so probabilities must be true |
| Explainability | **SHAP** (TreeExplainer) | exact Shapley values; explanations sum to the prediction |
| Retrieval | **sentence-transformers**, **FAISS** | local 384-dim embeddings, exact nearest-neighbour search |
| Generation | **Ollama** (`llama3.2:3b`) | runs locally — no transaction data leaves the machine |
| Serving | **FastAPI**, **Uvicorn**, **Streamlit** | automatic validation and generated API docs |
| Quality | **pytest** (62 tests) | run on synthetic fixtures — no dataset needed |

---

## Repository layout

```
src/risklens/
  config.py              project-root discovery, typed config from YAML
  data/
    dtypes.py            memory-aware dtype planning at parse time
    validate.py          seven data-contract assertions
    ingest.py            load → validate → join → verify → persist
    split.py             temporal split with embargo  ◀── the firewall
  eda/
    profile.py           missingness, correlated blocks, temporal drift
    stats.py             chi-square, Cramér's V, Mann-Whitney, Cliff's δ, PSI
    plots.py             seven decision-relevant figures
  features/
    build.py             deterministic features + FrequencyEncoder
    entity.py            causal entity aggregates (backward-only windows)
  models/
    train.py             baseline pipeline, XGBoost, imbalance handling
    evaluate.py          metrics, threshold strategies, cost model
    calibrate.py         isotonic / Platt, ECE, reliability tables
    explain.py           SHAP reason codes and a leakage audit
    unsupervised.py      IsolationForest, fraud typology clustering
  genai/
    narratives.py        scored transaction + SHAP → analyst case narrative
    search.py            embeddings, FAISS index, document chunking
    rag.py               retrieve → augment → generate, groundedness check
    agent.py             tool-using investigation copilot
  api/app.py             FastAPI scoring and investigation service

app/streamlit_app.py     analyst console
scripts/                 CLI entry points, in dependency order
tests/                   62 tests, synthetic fixtures, no dataset required
configs/data.yaml        paths, schema, data contract, split policy
corpus/policies/         synthetic fraud-operations policy documents (see note)
notebooks/               generated and executed; outputs embedded
reports/                 result artefacts — see below
```

---

## Reports: what they are and how they are made

`reports/` holds the **machine-readable output of every pipeline stage**. These
are not decoration — three of them are read by running code.

| File | Written by | Read by |
|---|---|---|
| `stage01_ingest_manifest.json` | `run_ingest.py` | provenance: SHA-256 of every input file |
| `stage02_*.csv` | `run_eda.py` | missingness, statistical tests, drift, V-block structure |
| `stage03_split_summary.json` | `run_eda.py` | exact split boundaries |
| `stage04_05_model_results.json` | `run_train.py` | **the API and the UI read the operating threshold from here** |
| `stage05_evaluation.json` | `run_eval.py` | the final test result |
| `stage05b_calibration.json` | `run_calibrate.py` | reliability tables before and after |
| `stage07_shap_importance.csv` | `run_genai.py` | feature importance; the notebook charts it |
| `stage06_09_genai_results.json` | `run_genai.py`, `run_llm.py` | typologies, RAG answers, copilot trace |
| `figures/*.png` | `run_eda.py` | seven analysis charts |

**Why they are committed.** Two reasons.

**Provenance.** `stage01_ingest_manifest.json` records the SHA-256 of each raw
input. The dataset itself is 678 MB and is not committed — it is regenerated by
`scripts/download_data.py`. Reproducibility therefore comes from the
*fingerprint* plus the script, not from storing the bytes. If a re-run produces
a different hash, results are not comparable and you find out immediately.

**They are the source of truth for every number.** The documentation is
*generated* from these files rather than typed by hand, so a written figure
cannot drift from what the code actually produced.

---

## Not included in this repository

Three things are deliberately absent, and the pipeline tells you clearly when
it needs them.

**The dataset** (678 MB). Public, but the competition licence restricts
redistribution. `scripts/download_data.py` fetches it, and
`reports/stage01_ingest_manifest.json` carries the SHA-256 of each file so
results stay traceable.

**Trained artefacts** (`models/`, `indexes/`). Regenerable binaries. Git
stores every version of a binary in full, so committing them would bloat
history permanently for no benefit.

## The policy corpus is synthetic

`corpus/policies/` contains six documents describing risk bands, card-not-present
controls, chargeback handling, account-takeover response, alert triage and model
governance. **They were written for this project.** They are not real policies
from any institution, and each file says so in its header.

They exist because retrieval needs a corpus and the IEEE-CIS dataset contains no
free text. Writing them rather than indexing real regulatory material was a
deliberate trade: external guidance would not reference this system's own risk
bands or thresholds, so the copilot would retrieve policy that did not describe
the system it was advising on.

The trade-off is worth stating: retrieving from documents written for the same
project is a weaker demonstration than indexing genuine regulatory text. The
loader reads any markdown files in that folder, so real documents can be dropped
in without code changes.

## Running it

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

python -m pytest                          # 62 tests, no dataset needed
```

The dataset requires a Kaggle account and one-time acceptance of the
[competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules).

```powershell
python scripts\download_data.py           # fetch train_transaction + train_identity

python scripts\run_ingest.py              # ingestion       → interim Parquet
python scripts\run_eda.py                 # EDA + split     → tables + figures
python scripts\run_train.py --fast        # features + models        (~30 min)
python scripts\run_all_downstream.py      # everything downstream    (~20 min)
```

`run_all_downstream.py` runs the serving schema export, evaluation,
calibration, SHAP, retrieval, the copilot, both notebooks and the result
summary — in dependency order, stopping on the first failure rather than
writing artefacts that disagree with each other.

```powershell
uvicorn risklens.api.app:app --port 8000              # API → localhost:8000/docs
streamlit run app\streamlit_app.py --server.port 8502 # UI  → localhost:8502
```

> **Resources.** ~1.9 GB disk for the dataset and derived Parquet. Training
> peaks around 4.8 GB of RAM.

---

## Design decisions worth explaining

| Decision | Reasoning | Rejected alternative |
|---|---|---|
| **LEFT join** on identity | Coverage is 24.4%; an inner join would delete 75.6% of rows and change the population being modelled | INNER join |
| **float32 features, float64 money** | Halves memory, but ~7 significant digits compounds error through the sums and ratios in feature engineering | uniform downcasting |
| **Parquet** | 8.8× smaller, 40× faster, preserves dtypes | CSV |
| **Temporal split with embargo** | Fraud is bursty — one compromised card produces near-identical transactions minutes apart, which a random split would place on both sides | `train_test_split(shuffle=True)` |
| **Class weighting** | Preserves the base rate, so predicted probabilities stay meaningful | SMOTE |
| **Causal entity aggregates** | Backward-only windows work in production; aggregating over train+test does not | transductive aggregation |
| **Raw entity IDs excluded** | 217,850 levels let the model memorise which customers had been defrauded rather than learn fraud behaviour | keeping them for a higher score |
| **PR-AUC over ROC-AUC** | ROC-AUC is optimistically biased under heavy imbalance | accuracy, ROC-AUC |
| **Cost-derived threshold** | A false negative costs the transaction amount; a false positive costs review time. Asymmetric, and one side varies | threshold 0.5, maximise F1 |
| **Local LLM** | No transaction data leaves the machine | hosted API |

---

## Scope

RiskLens is a fraud-detection system with an analyst-facing assistant layered
on top. The machine-learning pipeline stands on its own; the retrieval and
agent components consume it rather than replace it.

No distributed compute — 590,540 rows fit in 928 MB, so Spark would be slower
than pandas. No transformers beyond sentence embeddings for semantic search.
Nothing is included without a justification from the data or the problem.

---

## Data

IEEE-CIS Fraud Detection, Vesta Corporation, 2019 — a public Kaggle benchmark.
Features are anonymised: `V1`–`V339` have no published meaning, addresses are
coded integers, and email fields are domains only. The competition licence
permits academic and non-commercial use; the dataset is not redistributed here.
