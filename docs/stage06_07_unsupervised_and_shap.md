# Stages 6 & 7 — Unsupervised Learning and Explainability

**Every term defined in plain language, with a tiny worked example.**

---

# Stage 6 — Unsupervised Learning

## Why bother, when we already have a supervised model?

The supervised model can only recognise fraud that **resembles fraud it was
trained on.** In an adversarial domain that's a real weakness — criminals
change methods precisely to avoid known patterns.

**New term — Supervised learning:** you have labelled examples (this was
fraud, this wasn't) and the model learns the mapping.

**New term — Unsupervised learning:** you have **no labels**. The model finds
structure on its own.

Two *different* jobs here, and people often confuse them:

| | Anomaly detection | Clustering |
|---|---|---|
| **Question** | "Is this unlike anything I've seen?" | "What *kinds* of fraud are there?" |
| **Uses labels?** | No | Only to select the fraud rows |
| **Purpose** | Catch **unknown-unknowns** | **Describe** what we already caught |
| **Output** | An anomaly score | Named typologies |

---

## Anomaly detection — Isolation Forest

### How it works, intuitively

Build random trees by repeatedly picking a random feature and a random split
point. Then ask: **how many splits does it take to isolate this row alone?**

**Tiny example.** Ages: `25, 26, 27, 28, 29, 95`

```
Isolating 95:
   split at 60  →  {95} alone.                      DONE in 1 split.

Isolating 27:
   split at 60  →  {25,26,27,28,29}
   split at 26  →  {27,28,29}
   split at 28  →  {27}                             took 3 splits.
```

**Outliers get isolated quickly** because they sit in sparse regions. So a
**short average path length = anomalous.** That's the entire algorithm.

### Why this and not alternatives

| Alternative | Why rejected |
|---|---|
| One-class SVM | Roughly **quadratic** in rows — would not finish on 438k |
| Local Outlier Factor | Needs a distance metric; suffers the curse of dimensionality across 500 features |
| **Isolation Forest** ✅ | **Linear time**, no distance metric, no scaling needed, handles mixed scales |

**New term — Curse of dimensionality:** in high dimensions, all points become
roughly equidistant, so "nearest neighbour" stops being meaningful.

**Tiny example.** In 1D, points 1 and 2 are clearly closer than 1 and 100. In
500D with random values, almost every pair of points has a similar distance —
distance stops discriminating.

### The crucial property: it never sees a label

`fit(X)` — no `y`. That's the point. It must be able to flag a fraud type
**nobody has labelled yet**.

### How to judge it fairly

It will be **much** worse than the supervised model. That is **not a failure** —
the supervised model had 15,364 labelled examples; this had zero.

The right question: **does it beat random?** A random scorer gets PR-AUC = the
base rate = 0.035. If Isolation Forest beats that, fraud genuinely *is*
anomalous in feature space, which justifies keeping it as a safety net.

---

## Clustering — fraud typologies

### What we cluster, and why it matters

We cluster **only the confirmed fraud rows**, not the whole dataset.

**Why?** Clustering everything would just rediscover the majority class ("here
are the 96.5% normal ones"). We already know which rows are fraud. We're asking
a different question: **what kinds of fraud are there?**

### K-Means, briefly

**New term — K-Means:** pick k centre points, assign every row to its nearest
centre, move each centre to the average of its members, repeat until stable.

**Tiny example** with k=2 on transaction amounts `£5, £8, £10, £500, £520`:
```
Round 1: centres at £5 and £520
         → groups {5, 8, 10} and {500, 520}
Round 2: centres move to £7.67 and £510
         → same groups. Stable. Done.
```

### ⚠️ Why scaling is REQUIRED here (unlike for trees)

K-Means minimises **Euclidean distance**. Without scaling:

```
TransactionAmt : 0 → 31,000     ← dominates completely
is_night       : 0 → 1          ← contributes ~nothing
```

The clustering would effectively become "amount buckets", ignoring everything
else. **Trees don't care about scale** (they only compare values); **K-Means
does**.

### Choosing k — the silhouette score

**New term — Silhouette score:** for each point, compare how close it is to its
own cluster versus the nearest *other* cluster. Ranges −1 to +1.

- Near **+1** → clusters are well separated
- Near **0** → clusters overlap; the grouping is arbitrary
- Negative → points are in the wrong cluster

It gives a principled way to pick k instead of guessing.

> **Note:** silhouette is O(n²) in memory because it needs pairwise distances.
> We compute it on a 4,000-row subsample; running it on 15,000 fraud rows
> would allocate a 15,000 × 15,000 matrix.

### Making clusters actionable

A cluster called **"3"** is useless to an analyst. We compute how each cluster
differs from the fraud average **in standard deviations**, then use the
strongest deviations to auto-generate a label:

```
Typology 2: 3,140 cases (20.4%), avg amount £58
   signature: high C1, low D15, high card1_freq, low amt_log
   → reads as: many related addresses, recent prior activity,
     common card, small amounts  =  card testing
```

**This is what turns clustering from decorative into useful** — and it gives
the Stage 9 copilot something meaningful to cite.

---

# Stage 7 — Explainability with SHAP

## Why explainability is not optional

Three separate reasons, and only one is technical:

| Reason | Why it forces the issue |
|---|---|
| **Regulatory** | Under GDPR and financial regulation, a customer declined by an automated system can demand to know why. *"The gradient boosting model said so"* is not an answer. |
| **Operational** | An analyst receiving an alert needs to know **which** features triggered it, or they can't investigate. |
| **Debugging** | The **fastest way to find leakage** is to look at what the model considers important. |

---

## What SHAP actually is

**SHAP = SHapley Additive exPlanations.**

It borrows the **Shapley value** from cooperative game theory.

### The game theory origin, with a tiny example

Three people run a lemonade stand together and earn **£120**. How much did each
contribute? Shapley's answer: look at **every possible order** in which they
could have joined, and average each person's marginal contribution.

```
Alice alone:        £30      → Alice added £30
Alice + Bob:        £70      → Bob added £40
Alice + Bob + Carol:£120     → Carol added £50

But try another order...
Carol alone:        £20      → Carol added £20
Carol + Alice:      £80      → Alice added £60
...

Average across ALL orderings → each person's fair share.
```

**In SHAP, the "players" are features and the "payout" is the prediction.**

Each feature gets credit for how much it moved this prediction away from the
average, averaged over every possible ordering of features.

## The property that makes SHAP trustworthy: additivity

$$\text{base value} + \sum \text{SHAP values} = \text{the actual prediction}$$

**Tiny example.**
```
base value (average prediction)     0.035
  + is_night          = 1            +0.08
  + id_04 missing     = 0            +0.31
  + card1_freq        = 0.0002       +0.15
  + amount            = £425         −0.02
  + everything else                  +0.03
                                    ------
  = predicted probability             0.58
```

The explanation **sums exactly to the model's output.** It cannot be hand-wavy,
and it cannot omit a contributing factor. That's what distinguishes SHAP from
"feature importance" heuristics.

---

## TreeExplainer vs KernelExplainer

| | KernelExplainer | **TreeExplainer** ✅ |
|---|---|---|
| Works on | Any model | Tree models only |
| Method | Approximates by sampling | **Exact**, polynomial time |
| Speed | Thousands of model calls per row | Milliseconds |

Exact Shapley values require checking every subset of features — 2^470 subsets
here, which is impossible. **TreeExplainer exploits the tree structure** to
compute the exact answer in polynomial time.

**Always use TreeExplainer when the model is a tree.** It is both faster *and*
more accurate.

---

## Global importance: why mean **absolute** SHAP

```python
importance = np.abs(shap_values).mean(axis=0)
```

**Why absolute?**

**Tiny example.** A feature pushes risk **+0.5** for half the transactions and
**−0.5** for the other half.

- Signed average: `(+0.5 − 0.5) / 2 = 0` → looks useless
- Absolute average: `(0.5 + 0.5) / 2 = 0.5` → correctly identified as important

We want **magnitude of influence**, not net direction.

## Why SHAP beats XGBoost's built-in `feature_importances_`

| | XGBoost "gain" | **SHAP** |
|---|---|---|
| Computed on | **Training** data | Whatever data you give it (we use validation) |
| Bias | Favours **high-cardinality** features (more chances to split) | No such bias |
| Consistency | If a model relies on a feature *more*, gain can go *down* | Guaranteed to go up |

**Tiny example of the cardinality bias.** `card1` has 12,452 distinct values;
`card6` has 2. `card1` gets thousands more opportunities to appear in a split,
so gain importance inflates it regardless of real predictive value.

---

## The leakage audit

```python
if one_feature_share > 0.35:  →  LEAKAGE SUSPECT
```

**The reasoning:** Stage 2 found **nothing** with an effect size above ~0.24.
Real fraud signal is **diffuse**. A single feature accounting for more than a
third of all SHAP magnitude usually means it encodes the answer, the split, or
the time period.

**Tiny example of what this catches.** Suppose you accidentally left
`TransactionDT` in the features. SHAP would show it dominating, because the
model discovered `DT > 13,000,000 → test set`. The audit flags it in seconds;
without it you'd ship a model that scores brilliantly offline and fails
completely in production.

> **This is the single most useful debugging tool in the project.**

---

## Reason codes — the analyst-facing output

Raw SHAP is `V257: +0.31`. An analyst cannot act on that. So we map to plain
English:

```
↑ missing payer identity data  (increases risk, impact +0.312)
↑ how common this device is    (increases risk, impact +0.154)
↑ an overnight transaction     (increases risk, impact +0.081)
↓ the card network             (decreases risk, impact −0.024)
```

This is the unit an analyst consumes, and in Stage 9 it's what the LLM turns
into a narrative — **grounded in SHAP, so it cannot invent a driver.**

---

# Interview Q&A

### Q1. Why add unsupervised methods when you have a supervised model?
The supervised model can only recognise fraud resembling its training data,
which is a real weakness in an adversarial domain. Isolation Forest never sees
a label, so it can flag transactions unlike anything seen before — including
attack types that didn't exist when the model was trained. It's a safety net
for unknown-unknowns, not a replacement.

### Q2. How does Isolation Forest work?
It builds random trees and measures how many random splits it takes to isolate
each point. Outliers sit in sparse regions and get isolated in very few splits,
so short average path length means anomalous. It's linear time, needs no
distance metric and no scaling, which is why it works on 438k rows and 500
features where a one-class SVM wouldn't finish.

### Q3. Your Isolation Forest performs far worse than XGBoost. Isn't it useless?
No — that's the expected result and it's the wrong comparison. XGBoost had
15,364 labelled examples; the detector had zero. The right benchmark is random,
which scores PR-AUC equal to the base rate of 0.035. Beating that tells me
fraud genuinely is anomalous in feature space, which justifies keeping it for
novel attack types.

### Q4. Why cluster only the fraud rows?
Clustering everything would just rediscover the majority class. I already know
which rows are fraud; the question I'm asking is what *kinds* of fraud exist, so
the population of interest is the fraud cases themselves.

### Q5. Why does K-Means need scaling when XGBoost doesn't?
K-Means minimises Euclidean distance, so a feature ranging to 31,000 completely
dominates one ranging to 1 — the clustering would degenerate into amount
buckets. Trees only compare values within a feature, never across features, so
scale is irrelevant to them.

### Q6. What is SHAP and why do you trust it?
It applies the Shapley value from cooperative game theory: features are players
and the prediction is the payout, so each feature gets credit for how much it
moved this prediction from the average, fairly averaged over all orderings. I
trust it because of additivity — base value plus the SHAP values equals the
actual prediction exactly. The explanation can't omit a factor or hand-wave.

### Q7. Why TreeExplainer rather than the generic explainer?
Exact Shapley values need every subset of features, which is 2^470 here.
TreeExplainer exploits the tree structure to compute the exact values in
polynomial time. KernelExplainer is model-agnostic but approximates by
sampling and needs thousands of model evaluations per row.

### Q8. Why mean absolute SHAP for global importance?
Because a feature that pushes risk strongly up for some transactions and
strongly down for others is important, but the signed values would cancel to
zero and hide it. I want magnitude of influence, not net direction.

### Q9. Why not just use XGBoost's `feature_importances_`?
Gain importance is computed on training data and is biased toward
high-cardinality features, which get more chances to split — `card1` has 12,452
levels versus `card6`'s two. SHAP has no such bias, is computed on validation
data, and is consistent: if a model relies on a feature more, its SHAP
importance cannot decrease.

### Q10. How do you use SHAP to detect leakage?
I check whether any single feature accounts for a disproportionate share of
total SHAP magnitude — I flag above 35%. Stage 2 established that real signal
here is diffuse, with nothing above an effect size of 0.24. A single dominant
feature almost always means it encodes the label, the split, or the time
period. It's the fastest leakage check I have.

---

# Common Mistakes

1. **Comparing an unsupervised detector to a supervised model** — compare to random
2. **Not scaling before K-Means** — clustering degenerates to the largest-range feature
3. **Clustering the whole dataset** to find fraud types — you rediscover the majority class
4. **Leaving clusters as numbers** — "cluster 3" is useless to an analyst
5. **Using signed mean SHAP** — cancels out bidirectional features
6. **Trusting XGBoost gain importance** — biased toward high-cardinality features
7. **Running silhouette on the full dataset** — O(n²) memory blowup
8. **Using KernelExplainer on a tree model** — orders of magnitude slower and approximate
9. **Celebrating one dominant feature** — suspect leakage before success
10. **Treating SHAP as causal** — it explains the *model*, not the world

> **Point 10 deserves emphasis.** SHAP tells you what the model used, not what
> *causes* fraud. Stage 2's confounder finding is the perfect illustration:
> identity-present predicts fraud, but it doesn't *cause* it — the channel does.

---

# Files

| File | Purpose |
|---|---|
| `src/risklens/models/unsupervised.py` | Isolation Forest, K-Means typologies, silhouette |
| `src/risklens/models/explain.py` | TreeExplainer, global importance, reason codes, leakage audit |
| `scripts/run_genai.py` | Runs Stages 6–9 end to end |
