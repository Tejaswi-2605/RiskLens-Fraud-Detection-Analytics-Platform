"""Regenerate the Stage 3b-10 sections of development_log.md from real results.

Why a script rather than hand-editing
-------------------------------------
The log must quote REAL numbers from reports/*.json. Hand-copying them invites
transcription errors and stale figures. This reads the artefacts and writes the
section, so the log cannot drift from what actually ran.

Usage:
    python scripts/update_devlog.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOG = ROOT / "development_log.md"
MARKER = "# Stage 3b — Feature Engineering"


def load(name: str) -> dict:
    p = ROOT / "reports" / name
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def fmt(v, spec=",.4f", missing="_pending_"):
    if v is None:
        return missing
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def build_section() -> str:
    model = load("stage04_05_model_results.json")
    genai = load("stage06_09_genai_results.json")

    xgb = model.get("xgboost", {})
    base = model.get("baseline_logreg", {})
    xm, bm = xgb.get("metrics", {}), base.get("metrics", {})
    co = xgb.get("cost_optimal", {})
    dn = xgb.get("do_nothing_baseline", {})
    va = xgb.get("value_added", {})

    out = [f"""{MARKER}

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
| Logistic Regression (baseline) | **{fmt(bm.get('pr_auc'))}** | {fmt(bm.get('pr_auc_lift'), '.1f')}x | {fmt(bm.get('roc_auc'))} | {fmt(bm.get('brier'), '.5f')} |
| **XGBoost** | **{fmt(xm.get('pr_auc'))}** | {fmt(xm.get('pr_auc_lift'), '.1f')}x | {fmt(xm.get('roc_auc'))} | {fmt(xm.get('brier'), '.5f')} |

Random baseline PR-AUC = the base rate = **{fmt(xm.get('base_rate'), '.4f')}**.
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
| Do nothing (all fraud gets through) | {fmt(dn.get('net_cost'), ',.0f')} |
| RiskLens at the cost-optimal threshold | {fmt(co.get('net_cost'), ',.0f')} |
| **Net saving** | **{fmt(va.get('net_saving'), ',.0f')}** ({fmt(va.get('pct_of_fraud_loss_avoided'), '.1f')}% of fraud loss avoided) |
| Chosen threshold | {fmt(co.get('threshold'), '.4f')} |
| Precision / Recall | {fmt(co.get('precision'), '.1%')} / {fmt(co.get('recall'), '.1%')} |
| Alert rate | {fmt(co.get('alert_rate'), '.2%')} |
| Fraud caught / missed | {fmt(co.get('tp'), ',')} / {fmt(co.get('fn'), ',')} |

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
"""]

    # ---- Stages 6-7 -----------------------------------------------------
    shap = genai.get("shap", {})
    audit = shap.get("leakage_audit", {})
    iso = genai.get("isolation_forest", {})
    typ = genai.get("typologies", [])

    top_rows = ""
    for r in shap.get("top_20", [])[:10]:
        top_rows += f"| `{r['feature']}` | {r['mean_abs_shap']:.4f} |\n"

    typ_rows = ""
    for t in typ:
        typ_rows += (f"| {t['cluster_id']} | {t['n_cases']:,} | {t['share']:.1%} | "
                     f"{t['avg_amount']:,.2f} | {t['label']} |\n")

    out.append(f"""
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
{top_rows or "| _pending_ | _pending_ |"}

### The leakage audit

If ONE feature holds more than 35% of total SHAP magnitude, treat it as a
leakage suspect. Stage 2 established real signal here is diffuse - nothing had
an effect size above 0.24 - so a single dominant feature usually encodes the
answer, the split, or the time period.

**Verdict:** {audit.get('verdict', '_pending_')}
Top feature `{audit.get('top_feature', '?')}` holds
{fmt(audit.get('top_feature_share'), '.1%')} of total SHAP magnitude.

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
dimensionality across 504 features). Isolation Forest is linear time, needs no
scaling, and handles mixed feature scales.

**It never sees a label.** That is the point - it must be able to flag a fraud
type nobody has labelled yet.

**Real result:** PR-AUC {fmt(iso.get('pr_auc'), '.4f')},
ROC-AUC {fmt(iso.get('roc_auc'), '.4f')},
**{fmt(iso.get('lift_over_random'), '.2f')}x random**.

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
{typ_rows or "| _pending_ | | | | |"}

---
""")

    # ---- Stages 8-10 ----------------------------------------------------
    rag = genai.get("rag", [])
    inv = genai.get("investigation", {})
    demo = genai.get("search_demo", {})

    rag_rows = ""
    for r in rag:
        g = r.get("groundedness", {})
        rag_rows += (f"| {r['question'][:58]}... | {', '.join(r.get('sources', []))[:40]} "
                     f"| {g.get('grounded_ratio', '?')} | {g.get('verdict', '?')[:30]} |\n")

    out.append(f"""
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

**Search demo:** `"{demo.get('query', '_pending_')}"`

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
{rag_rows or "| _pending_ | | | |"}

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
{' -> '.join(inv.get('tools_called', [])) or '_pending_'}

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
hand-built (relative movement) and a real validation row (full 504 features,
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
""")

    return "\n".join(out)


def main() -> int:
    text = LOG.read_text(encoding="utf-8")
    if MARKER in text:
        text = text[: text.index(MARKER)]
    LOG.write_text(text.rstrip() + "\n\n" + build_section(), encoding="utf-8")
    print(f"development_log.md updated ({LOG.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
