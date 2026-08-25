# Stages 3b, 4 & 5 — Features, Models & the Risk Engine

**Every term defined in plain language, with a tiny worked example.**

---

# Stage 3b — Feature Engineering

## The one rule that organises everything

Features come in **two kinds**, and confusing them is how leakage happens.

| Kind | Definition | Tiny example | Safe to compute before the split? |
|---|---|---|---|
| **Deterministic** (row-wise) | Computed from **one row alone** | `log(amount)`, `hour of day`, `is this value missing?` | ✅ **Yes** |
| **Fitted** (cross-row) | **Learns** something by looking at many rows | frequency encoding, mean imputation, scaling | ❌ **No — train only** |

**The test:** *"Could I compute this for a single transaction arriving at the
API, with no dataset available?"*

- `log(amount)` → yes, just maths on one number. **Deterministic.**
- `how often does this card appear?` → no, I need to have counted the whole
  training set. **Fitted.**

---

## The deterministic features we built

### 1. Amount transformations

```python
amt_log      = log1p(TransactionAmt)
amt_cents    = TransactionAmt - floor(TransactionAmt)
amt_is_round = (amt_cents == 0)
```

**Why `log1p` and not `log`?**
`log(0)` = −infinity, which breaks everything. `log1p(x)` computes `log(1+x)`,
so `log1p(0) = 0`. Safe.

**Why log at all?** Money is heavily **right-skewed** — most transactions are
small, a few are enormous.

**Tiny example.** Amounts: `£5, £10, £20, £50,000`.
- Raw: the £50,000 dominates any distance calculation
- Logged: `1.79, 2.40, 3.04, 10.82` — now they're comparable

Trees don't need this (they only care about *order*), but the logistic
regression baseline does.

**Why the cents?** Real retail prices cluster at `.00`, `.99`, `.95`. Stolen-card
testing and currency conversion produce *unusual* decimals like `£43.2871`.
This is a well-known signal in this dataset.

### 2. Cyclical time features

```python
hour      = (TransactionDT // 3600) % 24
dayofweek = (TransactionDT // 86400) % 7
is_night  = hour in 0..6
```

**Important subtlety.** `TransactionDT` is *seconds from an unknown origin* —
we cannot recover a real calendar date.

**But** the origin is **constant**. So `% 24` gives a *relative* hour: the
labels are shifted (our "hour 3" might really be 8am), but **the shape of the
daily cycle is real**. Stage 2 confirmed a genuine daily pattern.

### 3. Missingness indicators ⭐

```python
for col in [id_01, id_02, ..., DeviceType, dist1, ...]:
    df[f"{col}_isna"] = df[col].isna()

df["n_missing"] = df.isna().sum(axis=1)
```

**This is Stage 2's headline finding turned into features.** We measured a
fraud rate of **10.31% when `id_04` is present** vs **2.61% when missing** — a
4× difference.

**Why make it explicit when trees handle NaN anyway?** Three reasons:
1. The logistic-regression baseline *cannot* use NaN at all
2. It makes the effect visible in SHAP (Stage 7) instead of hidden
3. `n_missing` compactly summarises the 14 correlated V-blocks in one number

### 4. Email domain features

```python
P_emaildomain_provider = "gmail.com"    → "gmail"
P_emaildomain_suffix   = "live.com.mx"  → "mx"
email_domains_match    = (P_emaildomain == R_emaildomain)
```

**Why split?** `gmail.com`, `gmail.co.uk` and `gmail.fr` are the same provider.
Splitting reduces **cardinality** (number of distinct values) without losing
information — and the suffix separately captures region.

**New term — Cardinality:** how many distinct values a column has.
`ProductCD` has 5 (low cardinality). `card1` has ~17,000 (high cardinality).

**Why the match flag?** A payer email different from the recipient email is a
classic **account takeover** indicator.

---

## The fitted feature: Frequency Encoding

### What it does

Replace a category with **how often it appeared in training**.

**Tiny example.** 1,000 training transactions:

```
card1 = 13926  appeared 500 times  →  card1_freq = 0.500   (common)
card1 = 88771  appeared   2 times  →  card1_freq = 0.002   (rare)
card1 = 99999  never seen          →  card1_freq = 0.000   (unseen)
```

### Why it works for fraud

**Rare is suspicious.** A card ID seen once among 438,000 transactions behaves
very differently from one seen 5,000 times.

### Why not one-hot encoding?

**New term — One-hot encoding:** turn each category into its own 0/1 column.

**Tiny example** with 3 colours:
```
colour            colour_red  colour_blue  colour_green
red        →           1           0             0
blue       →           0           1             0
```

Fine for 3 colours. For `card1`'s **17,000** distinct values, you'd create
**17,000 columns** — mostly zeros, enormous memory, and severe overfitting.

Frequency encoding turns 17,000 categories into **one** informative number.

### ⚠️ Why it MUST be a fitted transformer

```python
# ✗ LEAKED
df["card1_freq"] = df["card1"].map(df["card1"].value_counts(normalize=True))
#                                  ^^^ counts over train AND test

# ✓ CORRECT
fe = FrequencyEncoder(columns=[...])
fe.fit(df[train_mask])     # counts from TRAIN only
df = fe.transform(df)      # applied everywhere
```

**Why the leak matters.** If a card appears 50 times in the test period, using
the combined count tells the model something about the *future* volume of that
card. That information doesn't exist at prediction time.

Making it a proper scikit-learn transformer means `fit` runs on train inside a
`Pipeline`, and val/test only ever see `transform`. **The framework enforces
the discipline instead of my memory.**

### Handling unseen categories

A card appearing only in test gets `0.0`. That isn't arbitrary — **"never
observed in training" IS the rarest possible case**, and rarity is exactly what
this feature encodes.

---

## ⚠️ Two columns the model is NOT allowed to see

```python
always_drop = {target, "TransactionID", "TransactionDT"}
```

**Why drop `TransactionDT`?** It's the timestamp — it *monotonically increases*.

**Tiny example.** Train covers `DT ∈ [0, 11M]`, test covers `DT > 13.5M`. A
tree learns:

```
if TransactionDT > 13,000,000:  ← "I'm in the test set"
```

It doesn't learn fraud, it learns **which period a row came from**. That's
**leakage through the index**.

**But we keep the derived parts** — `hour`, `dayofweek` — because those encode
*behaviour* (when people transact) rather than *position in time*.

`TransactionID` has the same problem: it's a meaningless surrogate key that
also happens to increase with time.

---

# Stage 4 — Modelling

## Why train two models

| Model | Role |
|---|---|
| **Logistic Regression** | The **baseline** — establishes the number to beat |
| **XGBoost** | The **candidate** |

> **If XGBoost can't clearly beat logistic regression, ship the logistic
> regression.** A model you can explain to a regulator has genuine value in a
> bank. Complexity you can't justify does not.

Having a baseline is also the only way to answer *"is 0.70 PR-AUC good?"* —
good compared to **what**?

---

## Model 1 — Logistic Regression

**New term — Logistic regression:** fits a straight-line (linear) combination
of features, then squashes it into a 0–1 probability using the sigmoid function.

$$P(\text{fraud}) = \frac{1}{1 + e^{-(w_1x_1 + w_2x_2 + \dots + b)}}$$

**Tiny example** with two features:
```
score = 0.5 × (is_night) + 2.0 × (id_04_isna) − 3.0
if score is large → sigmoid pushes probability toward 1
```

Each weight is directly readable: *"night-time adds 0.5 to the fraud score."*
That interpretability is the whole point.

### Why each preprocessing step exists

| Step | Why |
|---|---|
| **Median imputation** | LogReg cannot accept NaN at all. **Median** not mean, because our numerics are skewed — the mean of a skewed column sits where no real value is. Our `_isna` flags preserve the missingness information separately. |
| **StandardScaler** | See below |
| **OneHotEncoder** | LogReg needs numbers, not text |
| **class_weight="balanced"** | Handles the 3.5% imbalance |

**Why scaling matters — tiny example.**

```
TransactionAmt : ranges 0 → 31,000
is_night       : ranges 0 → 1
```

L2 regularisation penalises **large coefficients**. To have any effect,
`TransactionAmt` needs a *tiny* coefficient (because its values are huge), and
`is_night` needs a *large* one. The penalty therefore punishes `is_night`
unfairly.

**Scaling** rescales every feature to mean 0, standard deviation 1, so the
penalty is fair.

**New term — Regularisation:** a penalty on large coefficients that discourages
the model from over-relying on any single feature. Prevents overfitting.

**`min_frequency=0.01`** in the one-hot encoder pools rare categories into a
single "infrequent" column, so we don't create thousands of near-empty columns.

**`handle_unknown="infrequent_if_exist"`** means a category first seen at
scoring time doesn't crash the API in Stage 10.

---

## Model 2 — XGBoost

**New term — Gradient boosting:** build many small decision trees *in
sequence*, where each new tree tries to correct the errors of the ones before.

**Tiny example.**
```
Tree 1 predicts:  0.30    (truth is 1.0, so error = 0.70)
Tree 2 learns to predict that 0.70 error, outputs: 0.40
Running total:    0.70    (error now 0.30)
Tree 3 corrects that...   and so on
```

**vs Random Forest**, which builds trees *in parallel* and averages them.
Boosting is sequential and usually more accurate on tabular data.

### Why XGBoost is right for this problem

**1. It handles NaN natively — the single biggest reason.**

229 of our 434 columns are >50% missing. XGBoost *learns* which side of each
split missing values should go.

**Tiny example.**
```
split on: dist1 < 100 ?
    yes  → left
    no   → right
    NaN  → the model LEARNS whether left or right gives better separation
```

Logistic regression would force you to invent a value. XGBoost treats
"missing" as its own meaningful direction.

**2. Native categorical support** (`enable_categorical=True`) — no one-hot
explosion on 17,000 card values.

**3. It captures interactions automatically.**

Fraud is **conjunctive** — "new device **AND** unusual hour **AND** rare card".
A path down a tree *is* an interaction. Logistic regression can only add
effects up; it can't express "these three things together".

**4. Scale-invariant.** Trees only compare values, so no normalisation needed.

### The hyperparameters, and why these values

**New term — Hyperparameter:** a setting you choose *before* training, as
opposed to a weight the model learns during training.

| Parameter | Value | Why |
|---|---|---|
| `n_estimators` | 600 | Maximum trees; early stopping picks the real number |
| `learning_rate` | 0.05 | How much each tree contributes. Lower = slower but more accurate |
| `max_depth` | 6 | A tree can express interactions of up to 6 features. Deeper memorises |
| `subsample` | 0.8 | Each tree sees 80% of **rows** |
| `colsample_bytree` | 0.8 | Each tree sees 80% of **columns** |
| `min_child_weight` | 5 | A leaf must hold at least this much weight |
| `eval_metric` | `aucpr` | Optimise **PR-AUC**, not accuracy |

**Why subsample/colsample = 0.8?** This is **bagging inside boosting**. By
giving each tree a different random view, the trees make *different* mistakes,
and averaging different mistakes cancels noise. Strong regularisation.

**Why `min_child_weight=5`?** Prevents a leaf built from 2 fraud cases. Two
examples isn't a pattern, it's memorisation.

**New term — Early stopping:** keep adding trees while validation performance
improves; stop when it plateaus.

**Tiny example.**
```
tree 100 → val PR-AUC 0.60
tree 200 → val PR-AUC 0.68
tree 300 → val PR-AUC 0.71
tree 350 → val PR-AUC 0.71   ← no improvement
tree 400 → val PR-AUC 0.70   ← getting WORSE = overfitting
                               STOP, keep 300
```

This means we never have to guess the right number of trees.

---

## Handling class imbalance

### What we did: `scale_pos_weight`

```python
scale_pos_weight = negatives / positives ≈ 27.6
```

**What it means:** during training, one missed fraud is penalised as heavily as
**27.6** false alarms. It changes the **loss function**, not the data.

### What we did NOT do: SMOTE

**New term — SMOTE (Synthetic Minority Over-sampling Technique):** creates fake
minority examples by interpolating between real ones.

**Tiny example.**
```
Real fraud A: amount £100, hour 3
Real fraud B: amount £200, hour 5
SMOTE invents: amount £150, hour 4   ← a transaction that never happened
```

**Three reasons we rejected it here:**

1. **The synthetic rows are meaningless.** Fraud A and Fraud B were committed
   by different people using different methods. Averaging them produces a
   transaction that couldn't exist.
2. **It's expensive** on 438,000 × 530 data.
3. **It destroys calibration** — the most important reason. SMOTE changes the
   base rate from 3.5% to 50%, so the model's output probabilities no longer
   mean anything real. **Our risk engine multiplies probability by money**, so
   we need probabilities that are actually true.

> **Interview answer:** *"I used class weighting rather than SMOTE. SMOTE
> invents synthetic frauds by interpolating between real ones committed by
> different people with different methods, which produces transactions that
> couldn't exist. More importantly it changes the base rate, so predicted
> probabilities stop being calibrated — and my risk engine needs calibrated
> probabilities to compute expected loss in pounds."*

---

# Stage 5 — Evaluation & the Risk Engine

## Why accuracy is banned

With 3.5% fraud, predicting "never fraud" gives **96.5% accuracy** and catches
**zero** fraud.

## The metrics we actually use

### Precision and Recall

**Tiny example.** 100 transactions, 10 are fraud. Your model flags 20, and 8 of
those are genuinely fraud.

- **Precision** = 8/20 = **40%** — *"of what I flagged, how much was right?"*
- **Recall** = 8/10 = **80%** — *"of all the fraud, how much did I catch?"*

**The trade-off:** flag everything → recall 100%, precision 10%. Flag nothing →
precision undefined, recall 0%.

**Which matters more?** In fraud, **recall** protects money and **precision**
protects the analyst team's time and customer goodwill. The right balance is a
business decision — which is exactly what the risk engine settles.

### PR-AUC ⭐ (our headline metric)

**New term — PR-AUC (Precision-Recall Area Under Curve):** sweep every possible
threshold, plot precision against recall, measure the area underneath.

**Crucially: a random model scores PR-AUC = the base rate = 0.035.**

So a PR-AUC of 0.70 is **20× better than random**. We report that lift, because
0.70 means nothing without the baseline.

### ROC-AUC (reported, but not optimised)

**Definition:** probability that a random fraud ranks above a random legitimate
transaction.

**Why we don't optimise it:** it's **optimistically biased** under heavy
imbalance because it rewards ranking the easy 96.5% majority correctly.

**Tiny example.** A model can score ROC-AUC 0.95 and still have terrible
precision, because 5% of a 570,000-row majority is 28,500 false positives —
swamping 20,000 true frauds.

### Brier score (lower is better)

**Definition:** mean squared error of the predicted probabilities. Measures
**calibration**.

**New term — Calibration:** when the model says 30%, does fraud actually happen
30% of the time?

**Tiny example.** Take all transactions where the model said "0.30".
- If 30 out of 100 really are fraud → **well calibrated** ✓
- If 5 out of 100 are fraud → **badly calibrated**, the model is over-confident

**Why we need it:** the risk engine computes `probability × amount = expected
loss in pounds`. If probabilities are wrong, the pounds are wrong.

---

## Choosing a threshold — three approaches

### Approach 1: minimum precision

*"My analysts need at least 50% precision or they stop trusting the system."*

If precision is 20%, four of every five alerts is an innocent customer. The
team disengages and the model's value evaporates regardless of its PR-AUC.

### Approach 2: alert budget ⭐ most realistic

*"My team can review 500 alerts a day out of 50,000 transactions."*

That's a **1% alert rate**. We take the 99th percentile of scores as the
threshold. This is the constraint real fraud operations actually work under.

### Approach 3: minimise money ⭐ what a risk engine does

This is what makes RiskLens a **risk system** rather than a classification
exercise.

**The cost model:**

| Outcome | Cost |
|---|---|
| **False negative** (fraud let through) | **the transaction amount** — we refund the customer |
| **False positive** (good customer declined) | **~£15** — review time + friction |
| **True positive** (fraud caught) | we recover ~90% of the value |

**The key insight: the two costs are asymmetric AND one of them varies.**

**Tiny example.**
```
A £10 transaction with 60% fraud probability
    → block it?  Expected saving £6, cost of a false alarm £15.  DON'T BLOCK.

A £5,000 transaction with 20% fraud probability
    → block it?  Expected saving £1,000, cost of a false alarm £15.  BLOCK.
```

**A lower probability justifies blocking a larger amount.** This is why a
single global threshold on probability is suboptimal, and why banks think in
**expected loss**, not in probability.

We sweep 200 candidate thresholds and pick the one minimising total net cost,
then compare against a **"do nothing"** baseline (all fraud goes through) to
quantify the model's value **in currency**.

> **Interview answer:** *"I don't pick a threshold by maximising F1. A false
> negative costs the transaction amount; a false positive costs about £15 of
> review time. Those are asymmetric, and one varies per transaction. So I sweep
> thresholds against an explicit cost model and choose the one that minimises
> net loss, then report the saving against doing nothing. That converts a
> PR-AUC into a number the business actually cares about."*

---

# Interview Q&A

### Q1. Why did you build a logistic regression if you knew XGBoost would win?
To establish what "good" means. Without a baseline, a PR-AUC of 0.70 is an
uninterpretable number. And the comparison is a genuine decision, not a
formality — if XGBoost only matched the baseline, I'd ship the linear model,
because explainability to a regulator has real value in a bank.

### Q2. Why is XGBoost a good fit for this dataset specifically?
Mainly the missing data. 229 of 434 columns are more than half empty, and
XGBoost learns which side of each split missing values belong on rather than
requiring me to invent values. It also handles categorical features natively,
which matters when `card1` has 17,000 levels, and it captures interactions
automatically — fraud is conjunctive, and a tree path *is* an interaction.

### Q3. Why not SMOTE?
It invents synthetic frauds by interpolating between real ones committed by
different people using different methods, producing transactions that couldn't
exist. It's also expensive at this size. But the decisive reason is
calibration: SMOTE changes the base rate, so output probabilities stop being
meaningful — and my risk engine multiplies probability by money.

### Q4. Why PR-AUC over ROC-AUC?
ROC-AUC is optimistically biased under heavy imbalance because it rewards
ranking the easy 96.5% majority. PR-AUC only considers the positive class,
which is the one I care about. I report both, and I report PR-AUC as a *lift*
over the base rate, since a random model scores 0.035.

### Q5. How did you choose the decision threshold?
Not statistically — economically. A false negative costs the transaction
amount; a false positive costs roughly £15 in review time and friction. Those
are asymmetric, and the false-negative cost *varies per transaction*. So I
sweep thresholds against that cost model and pick the minimum-cost point, then
compare to a do-nothing baseline to state the saving in currency.

### Q6. What's leakage-safe about your feature engineering?
I separate deterministic row-wise features, which are safe anywhere, from
fitted cross-row features, which are fitted on train only. Frequency encoding
is a proper sklearn transformer inside a Pipeline, so `fit` structurally cannot
see validation or test. And I exclude `TransactionID` and `TransactionDT`
entirely, because both increase with time and a tree would use them to identify
which period a row came from — leakage through the index.

### Q7. You keep `hour` but drop `TransactionDT`. Why isn't that inconsistent?
`TransactionDT` is *position* in time — it monotonically separates train from
test, so it's a proxy for the split itself. `hour` is `DT mod 24`, which
destroys that ordering and keeps only *behaviour*: when in the day people
transact. One leaks, the other generalises.

### Q8. What is early stopping and why use it?
Keep adding trees while validation performance improves; stop when it plateaus.
It means I don't have to guess the number of trees — the validation set picks
it. It's also my main defence against overfitting in boosting, since boosting
will happily keep fitting training noise forever.

### Q9. What's calibration and why do you care?
Whether a predicted 30% actually corresponds to 30% real-world frequency. I
care because the risk engine computes expected loss as probability × amount. If
the probabilities are systematically over-confident, every pound figure is
wrong even if the *ranking* is perfect.

### Q10. Your model has 530 features. Isn't that too many?
Possibly, and Stage 2 gave me a principled way to reduce them: the 339
V-columns collapse to 14 distinct missingness patterns, so they carry far less
independent information than their count suggests. With more time I'd do
block-level selection and check whether PR-AUC holds with a fraction of the
features — a smaller model is cheaper to serve and easier to explain.

---

# Common Mistakes

1. **Fitting the encoder on all data** — the classic leak
2. **Keeping the raw timestamp as a feature** — leakage through the index
3. **SMOTE before the split** — synthetic rows leak across the boundary
4. **SMOTE on validation/test** — you must evaluate on the *real* distribution
5. **Optimising accuracy** — 96.5% free by predicting nothing
6. **Threshold = 0.5 by default** — it's arbitrary and almost always wrong
7. **No baseline model** — you can't tell whether your score is good
8. **Reporting PR-AUC without the base rate** — 0.70 is meaningless alone
9. **Ignoring calibration** — fine for ranking, fatal for money decisions
10. **One-hot encoding a 17,000-level column** — memory explosion + overfitting

---

# Files

| File | Purpose |
|---|---|
| `src/risklens/features/build.py` | Deterministic features + `FrequencyEncoder` |
| `src/risklens/models/train.py` | Baseline pipeline, XGBoost config, imbalance |
| `src/risklens/models/evaluate.py` | Metrics, threshold selection, `CostModel` |
| `scripts/run_train.py` | Stages 3b→5 orchestration |
