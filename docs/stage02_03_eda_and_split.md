# Stages 2 & 3 — EDA, Statistics, and the Temporal Split

**Every term is defined in plain language, with a tiny worked example.**
Read this before an interview. Real numbers from our actual run.

---

## Why these two stages are combined

You might expect EDA (exploring) to come before splitting (dividing the data).
**It's the opposite, and the reason is subtle but important.**

> If I explore the *whole* dataset, notice "outlook.com has high fraud", and
> then build a feature for it — my brain has now used information from the
> test period. The code never touched test data, but **I** did. The model
> inherits my knowledge of the future.

This is called **human-in-the-loop leakage**, and no assertion or unit test can
catch it, because it happened inside your head.

**The fix:** split first, by a fixed rule. Then explore only the training part.

```
❌ WRONG:  load → explore everything → pick features → split → train
✅ RIGHT:  load → split → explore TRAIN ONLY → pick features → train
```

---

# Part 1 — The Temporal Split

## What a "split" is

You divide your data into three groups:

| Group | Purpose | How often you use it |
|---|---|---|
| **Train** | The model learns from this | Constantly |
| **Validation** | You tune and choose using this | Many times |
| **Test** | The final honest score | **Exactly once, at the very end** |

**Tiny example.** You're studying for an exam with 100 practice questions.
- **Train** = 70 questions you study from
- **Validation** = 15 questions you use to check if your study method works
- **Test** = 15 questions you save, sealed, to estimate your real exam score

If you peek at the test questions while studying, your predicted score becomes
meaningless. Same with models.

## Why *temporal* and not random

**Random split** = shuffle all rows, take 70% at random.
**Temporal split** = sort by time, take the earliest 70%.

For fraud, random is **wrong**, for two reasons:

### Reason 1: you'd be predicting the past from the future

In production you only ever have the past. A random split trains on December
to predict June — a task that will never exist.

### Reason 2: bursty fraud creates near-duplicates

**Tiny example.** A criminal steals card `4532...` and makes 8 purchases in
10 minutes:

```
10:01  £50   fraud
10:03  £50   fraud
10:04  £52   fraud
10:07  £50   fraud     ← if THIS lands in test
10:08  £51   fraud        and the others in train...
```

The model doesn't *predict* the 10:07 transaction — it **memorises** its
almost-identical siblings. Your score looks brilliant and means nothing.

A temporal split keeps the whole burst on one side.

## What an "embargo" is

**Definition:** a gap of time between partitions where rows are **thrown away**
rather than assigned to either side.

**Why:** even a temporal split has a hard boundary. A burst straddling it still
splits across train and test.

**Tiny example** with a 1-day embargo and a boundary at day 127:

```
day 126.9  ──→  TRAIN
day 127.2  ──→  DELETED  (inside the embargo)
day 127.8  ──→  DELETED  (inside the embargo)
day 128.3  ──→  VALIDATION
```

Cost: we lost **6,509 rows (1.1%)**. Benefit: no burst can straddle a boundary.
Cheap insurance.

> **Interview term:** this is called **purging and embargo**, from Marcos López
> de Prado's *Advances in Financial Machine Learning*. Naming the source is a
> strong signal in a finance interview.

## Our actual split

```
train  438,125 rows (74.2%)  fraud 15,364 (3.507%)  127.4 days
val     73,096 rows (12.4%)  fraud  2,536 (3.469%)   26.3 days
test    73,910 rows (12.5%)  fraud  2,634 (3.564%)   26.3 days
dropped  6,509 rows (1.1%)   — embargo
```

### Why is it 74/12/12 and not the 70/15/15 we asked for?

**Because we split by *time*, not by *row count*.** Those are different things.

**Tiny example.** Imagine 100 transactions over 10 days, but they're not evenly
spread — the shop was busier early on:

```
days 1-7:   80 transactions   ← 70% of the TIME
days 8-10:  20 transactions   ← 30% of the TIME
```

"70% of the time" = day 7. But that captures **80% of the rows**, not 70%.

Our transaction volume was higher in the earlier months, so the earliest 70% of
the *calendar* holds 74.2% of the *transactions*.

**This is deliberate and correct.** A real deployment says "retrain on the last
6 months," not "retrain on the last 400,000 rows." We match reality.

### The good news in those numbers

Fraud rate is **3.51% / 3.47% / 3.56%** across the three partitions — nearly
identical. This means the split didn't accidentally create an easy or
impossible test set.

## The bug the tests caught

With `embargo_days: 0`, a row landing *exactly* on the boundary matched **both**
train and val:

```python
train = t <= train_end      # a row at t == train_end → TRUE
val   = t >= val_start      # val_start == train_end → also TRUE  ← in BOTH!
```

**Fix:** make lower bounds exclusive.

```python
train = t <= train_end
val   = t >  val_start      # now the boundary row is in train only
```

**Lesson:** interval boundaries are where off-by-one leakage lives. A test with
`embargo=0` found it in seconds; it would have been invisible in production.

---

# Part 2 — What EDA Found

## Finding 1: over half the dataset is missing

```
columns with NO missing values : 95 / 434
columns >50% missing           : 229 / 434
worst: id_24 (99.15%), id_07 (99.08%), id_08 (99.08%)
```

**229 of 434 columns are more than half empty.** That isn't a broken dataset —
it's what real financial data looks like. Different systems capture different
fields.

## Finding 2: the 339 V-columns are really only ~14 things

**What I did:** grouped columns that are missing on *exactly the same rows*.

**Result: 339 V-columns → 14 distinct missingness patterns.**

```
block 1: 46 columns, all 76.12% missing
block 2: 43 columns, all  0.00% missing
block 3: 32 columns, all  0.00% missing
block 5: 29 columns, all 84.72% missing
```

**What this means.** If 46 columns are missing on *precisely* the same rows,
they came from the same source. They are **highly redundant** — 46 columns
carrying perhaps one or two columns' worth of independent information.

**Tiny example.** Three columns: `height_cm`, `height_inches`, `height_feet`.
All missing for the same people. They're one fact in three costumes.

**Why it matters:**
- Feature selection should treat a **block** as one unit
- Imputing them independently is wrong — they move together
- A single "is block 5 present?" flag may beat all 29 of its columns

**New term — Multicollinearity:** when features are so correlated they carry
duplicate information. It doesn't hurt tree models much (XGBoost just picks
one), but it wrecks the *interpretability* of linear models: the model can't
decide which of three identical features deserves the credit, so it splits the
weight arbitrarily.

## Finding 3 ⚠️ — MY HYPOTHESIS WAS WRONG

This is the most valuable finding in the project. Be ready to tell this story.

**What I predicted in Stage 1:** *"Missing identity data suggests evasion —
fraudsters avoid leaving device fingerprints. So fraud rate should be HIGHER
when identity is missing."*

**What the data actually said:**

| Column | % missing | Fraud rate when **MISSING** | Fraud rate when **PRESENT** |
|---|---|---|---|
| `id_04` | 88.3% | **2.61%** | **10.31%** |
| `id_09` | 86.7% | 2.51% | 10.02% |
| `dist2` | 93.2% | 3.07% | 9.44% |
| `DeviceType` | 74.5% | 2.12% | 7.58% |

Baseline fraud rate: **3.51%**

**Fraud is ~4× HIGHER when identity data is PRESENT.** The exact opposite of my
prediction.

### Why? (the honest explanation)

Identity/device data gets captured for **online, card-not-present** transactions
— you need a browser and a device to fingerprint. In-store chip-and-PIN
purchases have no browser to record.

And card-not-present fraud is *far* more common than in-person fraud, because
the criminal doesn't need the physical card.

So the real chain is:

```
identity present → it's an ONLINE transaction → online is riskier → more fraud
```

**Identity presence is a proxy for the sales channel, not for evasiveness.**

**New term — Confounding variable:** a hidden third factor that causes the
relationship you observe.

**Tiny example.** Ice cream sales correlate with drownings. Ice cream doesn't
cause drowning — **summer** causes both. Summer is the confounder.

Here: *channel* (online vs in-store) is the confounder driving both identity
capture and fraud rate.

### What this proves

**The direction was wrong, but the decision was right — and more strongly than I
argued.** A 4× difference in fraud rate is enormous. Missingness is one of the
most predictive things in the dataset. An INNER join would have deleted it.

> **Interview gold.** Say this out loud:
> *"I hypothesised that missing identity data indicated evasion. The data showed
> the opposite — a 4× higher fraud rate when identity is present. Investigating,
> that's a channel effect: identity is captured for card-not-present online
> transactions, which are inherently riskier. So the LEFT join was even more
> justified than I'd argued, but my causal story was wrong. I'd want to control
> for ProductCD before drawing any conclusion about device fingerprints."*
>
> This shows three things at once: you form hypotheses, you let data overrule
> them, and you look for confounders. That's worth more than being right.

## Finding 4 ⚠️ — Transaction amount does NOT predict fraud

```
TransactionAmt   Mann-Whitney U   p = 0.775   Cliff's delta = 0.0014
                 → "no detectable difference"
```

Everyone assumes fraudsters spend more. **In this dataset, they don't** — the
distributions are statistically indistinguishable.

**Why:** smart fraudsters deliberately keep amounts *normal*. Large unusual
charges trigger alerts. This is called **"card testing"** — small purchases to
verify a stolen card works before selling it on.

**The lesson:** your intuition about fraud is not evidence. Test it.

## Finding 5: what actually predicts fraud

Ranked by **effect size**, not p-value:

| Feature | Test | Effect size | Meaning |
|---|---|---|---|
| `D15` | Mann-Whitney | δ = −0.240 | small but real |
| `C1` | Mann-Whitney | δ = +0.235 | small but real |
| `C13` | Mann-Whitney | δ = −0.218 | small but real |
| `id_31` (browser) | Chi-square | V = 0.185 | weak but usable |
| `ProductCD` | Chi-square | V = 0.163 | weak but usable |
| `DeviceType` | Chi-square | V = 0.139 | weak but usable |
| `TransactionAmt` | Mann-Whitney | δ = 0.001 | **nothing** |

**Note that nothing is strong.** The best is "small but real." That's normal and
important: fraud detection wins by **combining many weak signals**, not by
finding one magic feature. If you *do* find a magic feature, suspect leakage.

## Finding 6: no drift between train and test

All PSI values < 0.06 — everything "stable."

Good news: our test period looks distributionally like our training period, so a
model that works on train has a fair chance on test.

---

# Part 3 — The Statistics, Explained Simply

## p-value

**Definition:** the probability of seeing a difference this large *if there were
truly no difference*. Small p (< 0.05) = "this is probably real."

**Tiny example.** You flip a coin 10 times, get 7 heads. p ≈ 0.34 — could easily
be luck. You flip 1000 times, get 700 heads. p ≈ 0.000000001 — that coin is
rigged.

### ⚠️ The trap: p-values are useless for ranking at our size

Look at our results — nearly every p-value is `0.000000e+00`. With **438,125
rows**, even a microscopic difference is "significant."

**Tiny example.** Fraudsters spend £100.01 on average, legitimate users £100.00.
- With 100 rows: p = 0.9, not significant
- With 10,000,000 rows: p < 0.0001, **highly significant**

The difference is 1 penny. Statistically significant, practically worthless.

**This is why we rank by effect size instead.** An interviewer may well ask
about this.

## Effect size

**Definition:** *how big* the difference is, independent of sample size.

### Cramér's V (for categorical features)

Measures association strength between two categories, from 0 to 1.

$$V = \sqrt{\frac{\chi^2}{n \times \min(\text{rows}-1,\ \text{cols}-1)}}$$

| V | Meaning |
|---|---|
| < 0.1 | negligible |
| 0.1 – 0.3 | weak but usable |
| 0.3 – 0.5 | moderate |
| > 0.5 | strong |

**Tiny example.** Does browser predict fraud?

```
              fraud   legit
Chrome          100    9900     1.0%
Internet Expl.  400    4600     8.0%
```

Chi-square says "yes, related." Cramér's V says "and here's how strongly."
Our real `id_31` (browser) scored **V = 0.185** — weak but genuinely usable.

### Cliff's delta (for numeric features)

**Definition:** pick one random fraud and one random legit transaction. How
often is the fraud one bigger?

Ranges −1 to +1. **0 = the distributions overlap completely.**

| \|δ\| | Meaning |
|---|---|
| < 0.147 | negligible |
| 0.147 – 0.33 | small |
| 0.33 – 0.474 | medium |
| > 0.474 | large |

**Tiny example.** `TransactionAmt` scored δ = **0.0014**. Meaning: pick a random
fraud and a random legitimate transaction, and the fraud is bigger **50.07%** of
the time. That's a coin flip. **No signal.**

Compare `D15` at δ = −0.240 — a real, if modest, separation.

## Chi-square test

**What it does:** for categorical data, compares what you *observed* against
what you'd *expect* if there were no relationship.

**Tiny example.** 1000 transactions, 5% fraud overall, 200 on mobile.
- **Expected** mobile fraud if channel doesn't matter: 200 × 5% = **10**
- **Observed** mobile fraud: **40**

40 ≫ 10, so chi-square is large → channel and fraud are related.

## Mann-Whitney U test

**What it does:** for numeric data, tests whether one group tends to have
*higher values* than another.

### Why not a t-test?

A **t-test** compares **averages** and assumes a roughly bell-shaped
distribution.

**Tiny example of why that fails here.** Nine transactions of £10, one of
£1,000,000:

```
mean   = £100,009    ← describes NONE of the transactions
median =      £10    ← describes 9 of the 10
```

`TransactionAmt` is heavily **right-skewed** (many small, a few enormous), so
the mean is dragged around by outliers.

**Mann-Whitney** ignores the actual values and uses **ranks** — 1st, 2nd, 3rd
biggest. The £1,000,000 outlier is just "rank 10." Robust.

**New term — Non-parametric:** a test that makes no assumption about the *shape*
of the distribution. Safer for messy real-world money data.

## PSI (Population Stability Index)

**What it does:** measures how much a distribution has *moved* between two
periods. **This is the standard drift metric in banking** — a model risk team
will ask for it by name.

$$\text{PSI} = \sum_i (a_i - e_i) \times \ln\!\left(\frac{a_i}{e_i}\right)$$

where $e_i$ and $a_i$ are the fraction of the expected (old) and actual (new)
populations in bin $i$.

| PSI | Verdict |
|---|---|
| < 0.10 | stable |
| 0.10 – 0.25 | moderate shift — investigate |
| > 0.25 | major shift — retrain |

**Tiny example.** Transaction amounts, split into 3 bins:

```
              Training    Production
small (£0-50)    50%   →     20%
medium (£50-200) 30%   →     30%
large (£200+)    20%   →     50%
```

Customers shifted to bigger purchases. PSI would be high — your model is now
seeing a different population than it learned from.

**Our result:** all features < 0.06 = stable. No drift between our train and
test periods.

**Why it's important for the JD:** this is exactly the kind of production
monitoring a bank's model-risk function requires. Mentioning PSI unprompted
signals you understand models *after* deployment, not just before.

---

# Part 4 — Interview Questions & Answers

### Q1. Why did you split before doing EDA?
To prevent human-in-the-loop leakage. If I explore all the data and then choose
features based on what I saw, my choices encode knowledge of the test period —
and no test can detect that afterwards, because it happened in my head. So the
split is defined by a fixed rule first, and I only ever explored the training
partition.

### Q2. Why a temporal split and not random?
Two reasons. First, in production you only have the past — a random split
measures a task that will never exist. Second, fraud is bursty: one compromised
card generates many near-identical transactions minutes apart. A random split
puts siblings of the same burst in both train and test, so the model memorises
rather than generalises, and the score is fiction.

### Q3. What's an embargo and why use one?
A gap between partitions where rows are dropped rather than assigned. Even a
temporal split has a hard boundary, and a burst straddling it would leak. I used
one day, which cost 1.1% of rows. It's purging and embargo from López de Prado's
financial ML work.

### Q4. Your split came out 74/12/12, not 70/15/15. Is that a bug?
No — deliberate. I split by time quantile, not row count. Transaction volume was
higher in the earlier months, so the earliest 70% of the calendar contains 74% of
the rows. That matches how a real deployment works: you retrain on "the last six
months," not "the last 400,000 rows."

### Q5. Why rank features by effect size instead of p-value?
At 438,000 rows, almost every p-value collapses to zero — even a difference of
one penny becomes "significant." The p-value tells me a difference is real; the
effect size tells me whether it's big enough to matter. Only the second one
ranks features.

### Q6. Why Mann-Whitney rather than a t-test on transaction amount?
A t-test compares means and assumes near-normality. Transaction amounts are
heavily right-skewed — a handful of huge values drag the mean somewhere that
describes no actual transaction. Mann-Whitney is non-parametric: it works on
ranks, so it's robust to those outliers.

### Q7. What surprised you in the data?
Two things. Transaction amount turned out to have essentially no predictive power
— Cliff's delta of 0.001, which is a coin flip. That's consistent with card
testing, where fraudsters deliberately keep amounts unremarkable.

And I had it backwards on missingness. I predicted missing identity data would
indicate evasion. In fact fraud is about four times *higher* when identity is
present. That's a channel effect — identity is only captured for card-not-present
online transactions, which are inherently riskier. So the missingness is
enormously informative, which justified my LEFT join even more strongly, but my
causal explanation was wrong. I'd want to control for ProductCD before saying
anything about device fingerprints specifically.

### Q8. You found 339 V-columns collapse to 14 missingness patterns. So what?
It means they aren't 339 independent features. Columns missing on exactly the
same rows came from the same upstream source and are heavily redundant. Practical
consequences: feature selection should operate on blocks rather than columns,
imputation should treat a block as a unit, and a single "is this block present"
indicator may outperform all 29 columns in it.

### Q9. What is PSI and why did you compute it?
Population Stability Index — the standard drift metric in banking. It measures
how far a distribution has moved between a reference period and a current one.
Below 0.1 is stable, above 0.25 means retrain. I computed it between my train and
test partitions; everything came out under 0.06, so the periods are
distributionally comparable. In production it's what you'd monitor to know when
the model has gone stale.

### Q10. None of your features are strong predictors. Isn't that a problem?
No, it's expected, and it's reassuring. The best effect size was around 0.24 —
"small but real." Fraud detection works by combining many weak signals, which is
exactly what gradient boosting is good at. If I *had* found one overwhelmingly
predictive feature, my first assumption would be leakage, not luck.

---

# Common Mistakes This Stage Avoids

1. **Exploring before splitting** — makes you the leak
2. **Random split on time-ordered data** — the single most common fatal error
3. **Ranking features by p-value at large n** — everything is significant
4. **t-test on skewed money data** — the mean isn't the centre
5. **Dropping high-missingness columns automatically** — missingness was our
   strongest signal here
6. **Treating 339 V-columns as 339 independent features** — they're ~14
7. **Trusting intuition over measurement** — "fraudsters spend more" is false here
8. **Rationalising a surprising result backwards** — we found a confounder instead
9. **Category fraud rates without a volume floor** — 1 fraud in 3 transactions
   is 33% and pure noise; we required n ≥ 500
10. **No drift check** — you can't know if train and test are comparable

---

# Files

| File | Purpose |
|---|---|
| `src/risklens/data/split.py` | Temporal split + embargo + safety assertions |
| `src/risklens/eda/profile.py` | Missingness, V-blocks, temporal, missing-as-signal |
| `src/risklens/eda/stats.py` | Chi-square, Cramér's V, Mann-Whitney, Cliff's δ, PSI |
| `src/risklens/eda/plots.py` | 7 figures, each answering one decision-relevant question |
| `scripts/run_eda.py` | Split-then-explore orchestration |
| `tests/test_split.py` | 14 tests — caught the embargo boundary bug |
| `reports/stage02_*.csv` | All tables |
| `reports/figures/*.png` | All 7 figures |
