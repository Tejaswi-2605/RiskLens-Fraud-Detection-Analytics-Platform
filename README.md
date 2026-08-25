# RiskLens — Intelligent Financial Risk & Fraud Detection Platform

An end-to-end fraud detection system built on the **IEEE-CIS Fraud Detection**
dataset (public Kaggle benchmark, Vesta Corporation, 2019). *Not proprietary data.*

**590,540 transactions · 434 features · 3.499% fraud · 182 days**

---

## The problem

Given a card transaction, estimate the probability it is fraudulent, and turn
that probability into an auditable **approve / review / decline** decision under
an explicit cost model.

Two properties make this hard, and they drive every design choice:

1. **Severe class imbalance** — 1 fraud per 27.6 legitimate transactions.
   Accuracy is meaningless: predicting "never fraud" scores **96.5%**.
2. **Time dependence** — fraud drifts. Our measured weekly fraud rate swings
   **2.07% → 5.08%**. Any random train/test split leaks the future into the past.

---

## Stage status

| # | Stage | Status | Teaching doc |
|---|---|---|---|
| 0 | Project setup & reproducibility | ✅ | [stage01_ingestion.md](docs/stage01_ingestion.md) |
| 1 | Data ingestion | ✅ | [stage01_ingestion.md](docs/stage01_ingestion.md) |
| 2 | EDA + data quality + statistics | ✅ | [stage02_03_eda_and_split.md](docs/stage02_03_eda_and_split.md) |
| 3 | Temporal split + feature engineering | ✅ | [stage02_03_eda_and_split.md](docs/stage02_03_eda_and_split.md) · [stage03b_04_05_modelling.md](docs/stage03b_04_05_modelling.md) |
| 4 | Supervised modelling | ✅ | [stage03b_04_05_modelling.md](docs/stage03b_04_05_modelling.md) |
| 5 | Evaluation + risk engine + calibration | ✅ | [stage03b_04_05_modelling.md](docs/stage03b_04_05_modelling.md) |
| 6 | Unsupervised + fraud typologies | ✅ | [stage06_07_unsupervised_and_shap.md](docs/stage06_07_unsupervised_and_shap.md) |
| 7 | Explainability (SHAP) | ✅ | [stage06_07_unsupervised_and_shap.md](docs/stage06_07_unsupervised_and_shap.md) |
| 8 | NLP + semantic search + RAG | ✅ | [stage08_09_genai.md](docs/stage08_09_genai.md) |
| 9 | Agentic investigation copilot | ✅ | [stage08_09_genai.md](docs/stage08_09_genai.md) |
| 10 | FastAPI + Streamlit | ✅ | [stage10_deployment.md](docs/stage10_deployment.md) |
| 10 | PySpark | 📝 written, **not run** | [stage10_deployment.md](docs/stage10_deployment.md) |
| 10 | Docker | 📝 written, **not built** | [stage10_deployment.md](docs/stage10_deployment.md) |

**Not run, and why — stated rather than hidden:**

- **PySpark** needs a JVM, which isn't installed here. It is also, honestly,
  *unnecessary for this dataset*: 590,540 rows fit in 928 MB, so Spark would be
  slower than pandas. The script exists to answer the question that does
  matter — which parts of the pipeline survive when data outgrows one machine —
  and it says so explicitly in its own output.
- **Docker** was skipped on disk grounds (Docker Desktop plus the image would
  consume ~10 GB). The `Dockerfile` and `docker-compose.yml` are complete and
  commented; they have simply never been built.

## Headline results

Measured on the **test** partition, which was opened once, at a threshold
chosen on validation and applied unchanged.

| Metric | Value |
|---|---|
| PR-AUC | **0.4680** (13.1× random) |
| ROC-AUC | 0.8883 |
| Brier (calibrated) | 0.02564 |
| Precision / Recall | 28.9% / 60.0% |
| Alert rate | 7.40% |
| **Fraud loss avoided** | **40.1%** (£158,772) |

Baseline logistic regression scored PR-AUC **0.3137** (9.0× random), so the
gradient-boosted model is +49% relative — which is what justifies its
complexity.

**Calibration** (Platt, fitted on the earlier half of validation):
Brier 0.0575 → **0.0223** (61% better), ECE 0.134 → **0.0081** (16.5× better),
maximum predicted probability 0.9999 → 0.7735, with **zero rank inversions**.

## Three bugs found by running the thing

Each was caught by a result being *impossible* rather than by a test:

1. **"116.8% of fraud loss avoided"** — the cost function treated caught fraud
   as revenue rather than avoided loss, so the optimiser was rewarded for
   flagging everything. Corrected to 44.2%.
2. **Fraud is 4× higher when identity data is PRESENT** — the opposite of the
   stated hypothesis. A confounder: identity is only captured for
   card-not-present transactions, which are inherently riskier.
3. **The copilot reported the CRITICAL action as the HIGH one** — a 3B model
   shifted a row while reading a markdown table. Fixed architecturally: the
   band→action lookup is now code, not prose comprehension.

## Key findings so far

| # | Finding | Consequence |
|---|---|---|
| 1 | Join preserved all **590,540** rows | Counts and rates are trustworthy |
| 2 | **3.499%** fraud, 1 : 27.6 imbalance | Use PR-AUC, never accuracy |
| 3 | Identity coverage only **24.4%** | LEFT join essential — INNER would delete 75.6% |
| 4 | Fraud rate swings **2.07% → 5.08%** | Random split would be indefensible |
| 5 | ⚠️ Fraud **4× higher when identity is PRESENT** | Channel effect, not evasion — **my hypothesis was wrong** |
| 6 | ⚠️ `TransactionAmt` has **no** signal (δ = 0.001) | Consistent with card testing |
| 7 | 339 V-columns → **14** missingness patterns | Heavy redundancy; use blocks not columns |
| 8 | All PSI **< 0.06** | No drift; train and test are comparable |

Finding 5 is the most instructive: I predicted missing identity meant evasion.
The data said the opposite. The cause is a **confounder** — identity is only
captured for card-not-present online transactions, which are inherently
riskier. The LEFT join was more justified than I'd argued, but my causal story
was wrong.

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Get the data

Needs a Kaggle account and one-time acceptance of the
[competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules).

```powershell
python scripts\download_data.py       # needs ~/.kaggle/kaggle.json
```

Or download `train_transaction.csv` and `train_identity.csv` manually into
`data\raw\`.

> **Disk:** ~1.9 GB total. **RAM:** the pipeline peaks around 4.7 GB — close
> other applications before running the full training.

---

## Running the pipeline

```powershell
python -m pytest                          # 26 tests, no dataset needed
python scripts\run_ingest.py              # Stage 1  → interim Parquet
python scripts\run_eda.py                 # Stages 2-3 → tables + 7 figures
python scripts\run_train.py               # Stages 3b-5 → models + metrics
python scripts\build_notebooks.py         # generate + execute notebooks
```

Every script has `--help`. `run_train.py --sample 150000` iterates faster.

---

## Architecture

```
data/raw/*.csv                     immutable, gitignored, SHA-256 hashed
        ↓  Stage 1: load → validate → join → verify → sort
data/interim/*.parquet             77 MB, 40× faster to load than CSV
        ↓  Stage 3: TEMPORAL SPLIT  ← the leakage firewall
   train (74.2%) │ val (12.4%) │ test (12.5%)   + 1-day embargo
        ↓  Stages 3b-7: features → models → evaluation → SHAP
models/*.joblib
        ↓  Stages 8-9: narratives → embeddings → RAG → agent
        ↓  Stage 10: PySpark · FastAPI · Streamlit · Docker
```

**The one rule everything obeys:**

> **Reshape data freely. Never *learn* from data before the split.**

Joining, retyping, renaming and sorting are per-row and safe. Anything that
computes a statistic across rows — an imputation mean, a scaler, a frequency
encoding — is fitted on the training partition only, inside an sklearn
`Pipeline`, so the framework enforces it rather than my memory.

---

## Layout

```
configs/data.yaml            paths, schema, data contract, split policy
src/risklens/
  config.py                  root discovery + typed config
  data/dtypes.py             memory-efficient dtype planning
  data/validate.py           7 data-contract assertions
  data/ingest.py             load → validate → join → verify → persist
  data/split.py              temporal split + embargo  ← the firewall
  eda/profile.py             missingness, V-blocks, temporal, missing-as-signal
  eda/stats.py               chi-square, Cramér's V, Mann-Whitney, Cliff's δ, PSI
  eda/plots.py               7 decision-relevant figures
  features/build.py          deterministic features + FrequencyEncoder
  models/train.py            baseline + XGBoost + imbalance handling
  models/evaluate.py         metrics, thresholds, CostModel risk engine
  models/explain.py          SHAP reason codes + leakage audit
  models/unsupervised.py     IsolationForest + fraud typology clustering
scripts/                     CLI entry points
notebooks/                   generated + executed, outputs embedded
docs/                        per-stage teaching docs
reports/                     tables, figures, manifests
tests/                       26 tests on synthetic fixtures
```

---

## Design decisions worth defending

| Decision | Why | Rejected alternative |
|---|---|---|
| **LEFT join** | Identity coverage 24.4%; INNER deletes 75.6% of data and biases the population | INNER join |
| **float32 features, float64 money** | Halves memory; but ~7 sig-digits compounds error through sums and ratios | Uniform downcasting |
| **Parquet** | 8.8× smaller, 40× faster, keeps dtypes, read natively by Spark | CSV |
| **Temporal split + embargo** | Fraud is bursty and drifts; a random split is fiction | `train_test_split(shuffle=True)` |
| **Class weighting** | Preserves calibration, which the risk engine needs | SMOTE |
| **PR-AUC** | Only considers the positive class; ROC-AUC is optimistically biased under imbalance | Accuracy, ROC-AUC |
| **Cost-optimal threshold** | FN costs the amount, FP costs ~£15 — asymmetric and amount-varying | Threshold 0.5, max F1 |
| **Local LLM (Ollama)** | Zero cost, offline, no data leaves the machine | Hosted API |

---

## Non-goals

Not a RAG/LLM project. The GenAI layer (Stages 8–9) sits *on top of* the ML
system as an analyst-facing copilot; the ML system stands on its own without
it. No transformers beyond sentence embeddings for semantic search. No GNN.
No technology added without a justification from the data or the problem.
