# RiskLens

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.1-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-337AB7)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.46-0088CC)](https://shap.readthedocs.io/)
[![FAISS](https://img.shields.io/badge/FAISS-1.9-4267B2)](https://faiss.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

**Fraud risk scoring and investigation platform for card payments.**

Built on the IEEE-CIS Fraud Detection dataset — a public benchmark released by
Vesta Corporation in 2019. *Not proprietary data.*

`590,540 transactions` · `504 engineered features` · `3.5% fraud` · `182 days`

---

## The problem

Given a card transaction, estimate the probability that it is fraudulent, and
turn that probability into an auditable **approve / review / decline** decision
under an explicit cost model — then explain the decision in language a fraud
analyst can act on.

Two properties of the data make this hard, and they drive every design choice.

**Severe class imbalance.** One fraud per 27.6 legitimate transactions.
Accuracy is meaningless: a model that predicts "never fraud" scores 96.5% and
catches nothing.

**Time dependence.** Fraud patterns drift — the measured weekly fraud rate
swings between 2.07% and 5.08%. Any evaluation that shuffles the data leaks the
future into the past and produces a score that will not survive production.

---

## Results

Measured on a held-out **test period**, opened once, at a decision threshold
chosen on validation and applied unchanged.

| Metric                       | Value                                     |
| ---------------------------- | ----------------------------------------- |
| **PR-AUC**             | **0.5144** — 14.4× a random model |
| ROC-AUC                      | 0.8969                                    |
| Brier score (calibrated)     | 0.0238                                    |
| Expected Calibration Error   | 0.0067                                    |
| Precision / Recall           | 29.7% / 66.0%                             |
| **Fraud loss avoided** | **45.6%** — $180,354 over 26 days |
| Generalisation gap           | −0.043                                   |

A logistic-regression baseline scores 0.3137. A random model scores the base
rate, 0.0347 — which is why PR-AUC is reported here as a *lift* rather than as
a bare number.

### The threshold is a business decision, not a statistical one

A false negative costs the transaction amount; a false positive costs analyst
review time. Those are asymmetric, and one of them varies per transaction — so
the operating point is chosen by minimising expected loss, not by maximising a
metric.

The right point also depends on how many alerts a team can actually review:

| Alert budget    | Precision | Recall |
| --------------- | --------- | ------ |
| 0.5% of traffic | 94.0%     | 13.6%  |
| 1%              | 88.0%     | 25.4%  |
| 2%              | 71.1%     | 41.0%  |
| 5%              | 42.2%     | 60.8%  |

---

## How it works

```
    Raw transaction and identity data
                 │
                 ▼
    ┌────────────────────────────┐
    │  Ingestion                 │   validated join, typed storage,
    │                            │   cryptographic provenance
    └────────────┬───────────────┘
                 ▼
    ┌────────────────────────────┐
    │  Temporal split            │   train on the past, test on the future,
    │                            │   with an embargo between periods
    └────────────┬───────────────┘
                 ▼
    ┌────────────────────────────┐
    │  Feature engineering       │   row-wise signals, cyclical time,
    │                            │   explicit missingness, and causal
    │                            │   per-entity behaviour
    └────────────┬───────────────┘
                 ▼
    ┌────────────────────────────┐
    │  Modelling                 │   linear baseline, then gradient
    │                            │   boosting with class weighting
    └────────────┬───────────────┘
                 ▼
    ┌────────────────────────────┐
    │  Risk engine               │   probability calibration, a cost
    │                            │   model, and a decision threshold
    └────────────┬───────────────┘
                 ▼
    ┌────────────────────────────┐
    │  Explanation & retrieval   │   per-alert reason codes, fraud
    │                            │   typologies, semantic case search,
    │                            │   policy retrieval, analyst copilot
    └────────────┬───────────────┘
                 ▼
    ┌────────────────────────────┐
    │  Serving                   │   scoring API and analyst console
    └────────────────────────────┘
```

### The rule the whole pipeline obeys

> **Reshape data freely. Never *learn* from data before the split.**

Joining, retyping and sorting are per-row operations and carry no information
between rows. Anything that computes a statistic *across* rows — an imputation
median, a scaler, a frequency map, a calibration curve — is fitted on the
training partition only.

Per-entity features use **expanding, backward-only windows**: for each
transaction, statistics over only that entity's *earlier* activity. That is
causal by construction, so there is no fitting step that could contaminate
validation or test — and unlike aggregating across the whole dataset, it would
actually work in production, where tomorrow's transactions do not exist yet.

---

## Approach

**Detection.** Gradient-boosted trees over 504 features. Chosen because more
than half the raw columns are largely missing, and the model learns which side
of each split missing values belong on rather than requiring invented
placeholders.

**Imbalance.** Handled by weighting the loss function rather than synthesising
minority examples. Synthetic oversampling changes the base rate, which destroys
the calibration the risk engine depends on.

**Calibration.** Class weighting fixes ranking but inflates probabilities. A
monotonic calibrator restores honest probabilities without disturbing the
ranking, because expected loss is probability multiplied by money.

**Explanation.** Shapley values give per-alert reason codes that sum exactly to
the prediction, so an explanation cannot omit a contributing factor. The same
importance view doubles as a leakage audit: real fraud signal is diffuse, so a
single dominant feature is a warning rather than a success.

**Investigation support.** Alerts are turned into readable case narratives,
embedded, and made searchable by meaning rather than keyword — so an analyst
can ask whether a pattern has been seen before. A retrieval-augmented assistant
answers policy questions from a controlled document set and refuses questions
outside it. It advises; a human decides.

---

## Stack

| Layer          | Technology                                                                |
| -------------- | ------------------------------------------------------------------------- |
| Data           | pandas, PyArrow / Parquet                                                 |
| Modelling      | scikit-learn, XGBoost                                                     |
| Explainability | SHAP                                                                      |
| Retrieval      | sentence-transformers, FAISS                                              |
| Generation     | a locally hosted language model — no transaction data leaves the machine |
| Serving        | FastAPI, Uvicorn, Streamlit                                               |
| Quality        | pytest — 62 tests, runnable without the dataset                          |

---

## Design decisions

| Decision                                   | Reasoning                                                                                                                                     |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Keep transactions with no identity record  | Coverage is 24%; dropping them would discard three quarters of the data and change the population being modelled                              |
| Preserve full precision on monetary values | Reduced precision compounds error through the sums and ratios used in feature engineering                                                     |
| Columnar storage                           | An order of magnitude smaller and faster than flat files, and it preserves types                                                              |
| Chronological split with an embargo        | Fraud is bursty — one compromised card produces near-identical transactions minutes apart, which a random split would place on both sides    |
| Weight the loss rather than oversample     | Preserves the base rate, so predicted probabilities stay meaningful                                                                           |
| Backward-only entity aggregates            | Works in production; aggregating across the full dataset does not                                                                             |
| Exclude raw entity identifiers             | With hundreds of thousands of levels, the model memorises which customers were previously defrauded instead of learning what fraud looks like |
| Precision-recall over ROC                  | ROC is optimistically biased under heavy imbalance                                                                                            |
| Cost-derived threshold                     | The two error types have different and unequal costs, and one of them varies per transaction                                                  |
| Local language model                       | No transaction data leaves the machine                                                                                                        |


## Scope

RiskLens is a fraud-detection system with an analyst-facing assistant layered
on top. The machine-learning pipeline stands on its own; the retrieval and
assistant components consume it rather than replace it.

No distributed compute — the data fits comfortably in memory, so a cluster
framework would be slower than the single-machine implementation. Nothing is
included without a justification from the data or the problem.

The policy documents used by the retrieval layer are **synthetic**, written for
this project. They are not real policies of any institution, and each states so.

---

## Data

IEEE-CIS Fraud Detection, Vesta Corporation, 2019 — a public Kaggle benchmark.
Features are anonymised: the behavioural signals have no published meaning,
addresses are coded integers, and email fields are domains only. The
competition licence permits academic and non-commercial use; the dataset is not
redistributed here.
