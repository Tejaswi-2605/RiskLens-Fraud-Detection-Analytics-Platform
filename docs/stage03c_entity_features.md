# Stage 3c — Entity-Linkage Features

**Every term defined in plain language, with a tiny worked example.**

This was the single highest-leverage change to the model, and it's the one
where I deliberately did something *worse* than the competition winners — for
a reason worth being able to explain.

---

## The problem it solves

Up to now, the model saw each transaction **in isolation**. But fraud is a
pattern over an **entity** — one compromised card making several purchases in
quick succession.

**Tiny example.** A criminal steals card `4532…` and makes four purchases:

```
10:01  £50   at an online shop
10:03  £52   at another
10:04  £48   at a third
10:07  £51   at a fourth
```

Each row, viewed alone, is **completely unremarkable**. £50 is a normal
amount. An online shop is a normal merchant.

**The fraud is only visible in the *relationship between* the rows.** A
single-row feature set physically cannot express *"this is the fourth
transaction on this card in six minutes."*

---

# Step 1 — Normalise the `D` columns

## The problem with `D1`–`D15`

They mean *"days since some previous event."* Measured **relative to each
transaction** — so the same real-world event produces a **different number**
depending on when you look.

**Tiny example.** One card, two purchases ten days apart:

```
                    day    D1     meaning
Transaction A       100    30     "card first seen 30 days ago"
Transaction B       110    40     "card first seen 40 days ago"
```

Raw `D1` says **30** and **40** — they look like two unrelated situations. But
they describe the *same* fact: the card was first seen on day 70.

## The fix

```python
day = TransactionDT / 86400
D1n = floor(day − D1)        # the ABSOLUTE day the event occurred
```

```
Transaction A:  100 − 30 = 70
Transaction B:  110 − 40 = 70     ← IDENTICAL
```

A **drifting offset** becomes a **stable identity signal**.

> **Why `floor`?** The underlying quantity is whole days. Without rounding,
> float noise would split one entity into two.

---

# Step 2 — Build a pseudo-client UID

We have no customer ID. So we **infer** one:

```
uid  = card1 + "_" + addr1 + "_" + D1n
uid2 = uid   + "_" + card2 + "_" + card3
```

**The logic:** same card number, same billing region, same first-seen day ⇒
*very probably the same client*.

This is the **"magic feature"** from the IEEE-CIS competition.

## Two levels, because they trade off differently

| | Groups | Precision |
|---|---|---|
| `uid` (coarse) | more transactions → better statistics | more collisions between real clients |
| `uid2` (fine) | fewer per group | more precise, but many appear once and their stats are meaningless |

Let the model choose which to lean on.

## Our real numbers

```
590,540 transactions  →  217,850 distinct uid
                      →  219,998 distinct uid2
```

Average **2.7 transactions per entity**. Many appear once — which is why the
"prior" features are ~63% non-null.

## Missing values become a group, not a hole

```python
addr1 = NaN  →  "na"
```

Dropping those rows would silently discard data. And *"we don't know the
billing region"* is a **consistent, informative group** — Stage 2 already
proved missingness carries signal here.

---

# Step 3 ⭐ — Aggregate causally

**This is the interesting part, and where I diverge from the winners.**

## What the Kaggle winners did

They computed aggregates over **train AND test combined**.

That's **legal in a competition** — Kaggle hands you the test features (just
not the labels), so "how many times does this card appear overall?" is
computable.

## Why I didn't

**It's impossible in production.**

**Tiny example.** It's Tuesday. A transaction arrives on card `4532…`. To
compute *"this card appears 47 times in the dataset"*, you would need to know
about transactions that **haven't happened yet**.

**New term — Transductive learning:** using the *features* (not labels) of the
test set during training. Fine for a fixed benchmark. Meaningless for a live
system, where "the test set" is tomorrow.

## What I did instead: expanding, backward-only windows

For each transaction, aggregate over **only that entity's earlier
transactions**.

```
Card 1's history:  £100, £200, £300, £1000

row 1  (£100)   prior = {}                → NaN
row 2  (£200)   prior = {100}             → mean 100
row 3  (£300)   prior = {100, 200}        → mean 150
row 4  (£1000)  prior = {100, 200, 300}   → mean 200
```

**Causal by construction.** A row can only ever see rows that came before it,
so train/test contamination is **not possible** — there's no `fit` to leak.

## The six features, per entity view

| Feature | What it captures |
|---|---|
| `count_prior` | how many times seen **before now** |
| `amt_mean_prior` | running average spend |
| `amt_std_prior` | running spread |
| **`amt_ratio`** | this amount ÷ running mean → *"unusual **for this customer**"* |
| **`secs_since_last`** | velocity — the card-testing signal |
| `txns_last_day` | burst detector |

### Why `amt_ratio` matters conceptually

Stage 2 found **raw `TransactionAmt` has no predictive power at all**
(Cliff's δ = 0.0014 — a coin flip).

**Here's why that's not the whole story:** £500 is unremarkable for one client
and extraordinary for another. Only a **per-entity** view can tell them apart.
A global amount threshold structurally cannot.

---

## ⚠️ The one line that makes it correct

```python
csum = g[amount_col].cumsum() - amt     # ← subtract the current row
```

**Without that subtraction, every transaction contributes to the average it's
being compared against.**

**Tiny example of the self-leak:**

```
                    WITH self          WITHOUT self (correct)
row 2 (£200)        mean = 150         mean = 100
                    ratio = 1.33       ratio = 2.00
```

The self-inclusive version **dilutes** the very anomaly you're trying to
detect — and, worse, it means a fraudulent transaction partly explains itself.

> **This is a leak that makes validation scores go UP, not down.** Nothing
> alerts you. It has to be a test, not a careful comment.

## Why `cumsum()` and not `.expanding()`

`groupby(...).expanding().mean()` is correct but **crawls** at 590k rows.
`cumsum()` computes the same thing in one vectorised pass — **86 seconds** for
all 36 features.

---

# The measured signal

Fraud rate across quintiles of each new feature (baseline 3.50%):

| Feature | Q1 → Q5 | Spread |
|---|---|---|
| **`uid_secs_since_last`** | **7.02% → 1.07%** | **6.5×** ⭐ |
| `uid_count_prior` | 2.94% → 4.78% | 1.6× |
| `card1_txns_last_day` | 3.09% → 3.23% | 1.4× |
| `uid_amt_ratio` | 3.63% → 4.07% | 1.1× |

## Reading the winner

**Fast repeats on one entity are 6.5× more fraudulent than slow ones.**

That is **exactly** the card-testing pattern the policy corpus describes:
small, rapid purchases to verify a stolen card is live.

> **For context: this is stronger than anything Stage 2 found univariately** —
> the best effect size there was 0.24, and every feature was "small but real."
> This one feature separates a 7% population from a 1% population.

---

---

# ⭐ The result — and the mistake I made getting there

## The three-way comparison

I ran the model three ways. The comparison **is** the finding.

| Configuration | Val PR-AUC | Fraud loss avoided | Verdict |
|---|---|---|---|
| No entity features | 0.5232 | 37.4% | the baseline |
| **+ behavioural aggregates** | **0.5574** | **49.7%** | ✅ **the honest number** |
| + raw entity IDs (`uid`, `uid2`) | 0.6577 | — | ❌ inflated, and unshippable |

## What went wrong

My first version passed the raw `uid` and `uid2` **categoricals** to the model,
not just the aggregates built from them. PR-AUC jumped to **0.6577 — a 26%
gain**, and I nearly reported it.

**Two things made me suspicious:**

**1. The round-0 score.** A single tree is one split:

```
round 0, no entity features   0.2835
round 0, + aggregates         0.2671
round 0, + raw entity IDs     0.4235   ← one split reached 0.42
```

Behaviour has to accumulate across many trees. **Identity lookup pays off on
the first one.** That signature is memorisation.

**2. It crashed.**

```
xgboost/training.py → bst.copy() → __getstate__()
XGBoostError: bad allocation
```

Every tree node splitting on `uid` stores a bitset over **217,850 categories**
— roughly 27 KB *per split node*. Across 300 trees the model became too large
to serialise. **You cannot deploy a model you cannot save.**

## Why memorising entities is rewarded here

The dataset's labelling rule propagates fraud **forward** across transactions
linked by card, email or billing address. So "this entity was defrauded before"
is *mechanically* predictive of "this entity is labelled fraud now."

The model isn't detecting fraud. It's **rediscovering the labelling rule**.

**It isn't pure leakage** — in production you do see repeat customers, and
blocklisting a previously compromised card is legitimate. But it inflates the
offline number relative to performance on **new** customers, which is what you
actually need.

## A second hypothesis I got wrong

The honest gain was smaller than I expected, so I guessed *"the `C1`–`C14`
columns are documented as counts of related addresses and entities, so mine
are redundant."*

**I measured it instead of asserting it:**

```
median max-correlation between my 15 features and any C column:  0.030
```

**Essentially zero.** Every feature carries genuinely new information. My
explanation was wrong.

**The real reason** is visible in the data: **2.7 transactions per entity** on
average, and the prior-statistics are only **63% non-null**. There is very
little history to aggregate over.

> That's a better diagnosis than my guess, and it has a clear implication:
> **with a longer transaction history these features would be considerably
> stronger.** Which is exactly the situation a real bank is in.

## What I would say about this

> *"Entity features looked like a 26% improvement. Isolating the two mechanisms
> showed about 6.5% came from behavioural signal — velocity, running spend,
> burst counts — and the rest from the model memorising which of 217,850
> customers had previously been defrauded. This dataset propagates fraud labels
> across linked cards and addresses, so that memorisation is rewarded offline
> and would collapse on new customers. It also made the model too large to
> serialise. I dropped the raw IDs and quote 0.5574."*
>
> *"The behavioural gain looked modest, so I checked whether it was redundant
> with Vesta's existing count columns. Correlation was 0.03 — it wasn't. The
> real constraint is that entities average 2.7 transactions here, so there is
> almost no history to aggregate. With a longer window these features would do
> considerably more."*


# Interview Q&A

### Q1. What was the single biggest improvement you made, and why did it work?
Entity-linkage features. The model saw each transaction in isolation, but
fraud is a pattern over an entity — one compromised card making several
purchases in minutes. No single-row feature can express that. I inferred a
client ID from card number, billing region and first-seen day, then computed
backward-looking aggregates per entity. Time-since-that-entity's-last-
transaction alone separates a 7% fraud population from a 1% one.

### Q2. Why normalise the D columns first?
They're "days since some event", measured relative to each transaction, so the
same real event gives different numbers over time — 30 today, 40 ten days
later. Subtracting from the day index gives the absolute day the event
occurred, which is stable. Without that, two transactions from one card don't
agree on anything and you can't link them.

### Q3. The winning Kaggle solutions did this differently. How, and why didn't you copy them?
They aggregated over train and test combined. That's legal in a competition —
you're given the test features — but impossible in production, because you
can't know a card's future transactions while scoring today's. I used
expanding backward-only windows instead. It gives less lift, and that's the
honest cost of a feature that would actually work in deployment.

### Q4. How do you know your aggregates don't leak?
Structurally, they can't — a row only ever sees rows before it, so there's no
`fit` step to contaminate. But I test it rather than assert it: the key test
appends a large *later* transaction and asserts that no *earlier* row's
features change. That matters because this class of leak raises validation
scores rather than lowering them, so nothing would alert me.

### Q5. What's the subtlest bug in this code?
Forgetting to exclude the current row from its own statistics. If you use
`cumsum()` without subtracting the current value, every transaction
contributes to the average it's compared against — which dilutes the anomaly
and means a fraudulent transaction partly explains itself. It looks like
signal and generalises to nothing.

### Q6. Stage 2 said transaction amount had no predictive power. Why build an amount feature now?
Because globally it doesn't, and per-entity it might. £500 is unremarkable for
one client and extraordinary for another; only a per-customer baseline can
distinguish them. In practice `amt_ratio` turned out weak here too — 1.1×
spread — but the reasoning is sound and it's a cheap feature.

### Q7. You have 217,850 entities over 590,540 rows. Isn't 2.7 transactions each too few?
It's thin, and it shows: the prior-statistics are only ~63% non-null, because
so many entities appear once. That's why I built two granularities — the
coarser `uid` groups more transactions at the cost of some collisions between
real clients. With a longer history the features would be considerably
stronger.

### Q8. Why `cumsum` rather than `expanding`?
Same result, vastly faster. `groupby().expanding().mean()` is
Python-loop-bound and crawls at 590k rows; cumulative arithmetic is one
vectorised pass — 86 seconds for all 36 features.

---

# Common Mistakes

1. **Aggregating over train + test** — the Kaggle habit; impossible in production
2. **Including the current row in its own statistics** — a self-leak that *raises* your score
3. **Not normalising `D` first** — un-normalised offsets can't link entities at all
4. **Dropping rows with a NaN key component** — "unknown region" is a valid group
5. **Only building one granularity** — coarse and fine trade off differently
6. **Using `.expanding()`** — correct but far too slow at this scale
7. **Trusting the features without measuring** — I checked the fraud-rate spread *before* spending 25 minutes retraining
8. **Assuming causality instead of testing it** — this leak makes scores better, so only a test catches it

---

# Files

| File | Purpose |
|---|---|
| `src/risklens/features/entity.py` | D-normalisation, UID construction, causal aggregates |
| `tests/test_entity.py` | 11 tests, including the no-future-leakage guarantee |
| `scripts/run_train.py` | `--no-entity` for A/B comparison |
