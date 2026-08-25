# Stage 1 — Data Ingestion

**Every term defined in plain language, with a tiny worked example.**
Real numbers from our actual run on 590,540 transactions.

---

## What this stage does, in one sentence

Turn two raw CSV files into **one trustworthy table** that every later stage
can rely on — and make it *impossible* for four specific disasters to happen
silently.

```
data/raw/train_transaction.csv  ─┐
                                  ├─→ [INGEST] ─→ transactions_joined.parquet
data/raw/train_identity.csv     ─┘                stage01_ingest_manifest.json
```

---

## Why this isn't just `pd.read_csv`

Four things go wrong silently if you don't guard them:

| Disaster | What you'd see | What actually happens |
|---|---|---|
| Join fan-out | Nothing. No error. | Every count and rate you report is wrong, forever |
| Memory blow-up | Your laptop freezes | Can't even load the data |
| Schema drift | Trains fine | Crashes in the live API 3 months later |
| Premature cleaning | Great scores! | **Data leakage** — model fails in production |

> A pipeline that **crashes** is annoying.
> A pipeline that **silently produces wrong data** is dangerous — it makes a
> plausible-looking model that denies real customers' transactions.

---

## Part 1 — The Join

### What a "join" is

Combining two tables using a shared column.

**Tiny example.** Two tables about the same purchases:

```
TRANSACTIONS                    IDENTITY
id | amount | fraud             id | device
---|--------|------             ---|--------
 1 |  £50   |  0                 1 | iPhone
 2 |  £30   |  0                 3 | Android
 3 | £900   |  1
 4 |  £20   |  0
```

Notice: identity has data for transactions 1 and 3 **only**. That's the whole
problem in miniature.

### LEFT JOIN vs INNER JOIN

**INNER JOIN** — keep only rows that exist in *both* tables:

```
id | amount | fraud | device
---|--------|-------|--------
 1 |  £50   |  0    | iPhone
 3 | £900   |  1    | Android
```
→ **2 rows.** Transactions 2 and 4 are **deleted**.

**LEFT JOIN** — keep *all* left rows, fill gaps with `NaN`:

```
id | amount | fraud | device
---|--------|-------|--------
 1 |  £50   |  0    | iPhone
 2 |  £30   |  0    | NaN     ← kept!
 3 | £900   |  1    | Android
 4 |  £20   |  0    | NaN     ← kept!
```
→ **4 rows.** Nothing lost.

### We chose LEFT. Here's why, with our real numbers.

Our identity coverage was **24.42%**. An INNER join would have deleted
**446,307 of 590,540 rows (75.6%)**.

**Three reasons that would be a disaster:**

**1. You throw away fraud examples you can't spare.**
Only 3.5% of rows are fraud. Deleting 75% of the data deletes ~75% of your
positive examples. With rare events, every one counts.

**2. You change the population you're modelling — "selection bias."**

**New term — Selection bias:** when your sample isn't representative of the
group you'll actually apply the model to.

**Tiny example.** You want to know the average height of everyone in a city.
You measure people leaving a basketball arena. Your answer is wrong — not
because you measured badly, but because you sampled the wrong population.

Here: transactions *with* device fingerprints are not a random 24% of all
transactions. Train on them, deploy on everything, and the model is
mismatched to reality.

**3. You delete the signal.**

**New term — Informative missingness:** when the *fact that something is
missing* tells you something.

**Tiny example.** A job application form with "previous salary" left blank.
That blank isn't nothing — it might mean the applicant is uncomfortable
answering. The blank carries information.

Stage 2 proved this dramatically for us — see below.

### The verification: proving the join didn't corrupt anything

**New term — Join fan-out (row multiplication):** if the right table has
duplicate keys, one left row matches multiple right rows, so the output has
*more* rows than the input.

**Tiny example.**

```
LEFT              RIGHT (id 1 appears twice!)
id | amount       id | device
---|-------       ---|-------
 1 |  £50          1 | iPhone
 2 |  £30          1 | iPad

RESULT — 3 rows from a 2-row input:
id | amount | device
---|--------|-------
 1 |  £50   | iPhone     ← the £50 is now counted TWICE
 1 |  £50   | iPad       ←
 2 |  £30   | NaN
```

Your total revenue is now wrong. Your fraud rate is wrong. **Nothing crashed.**

**Our three-layer guard:**

```python
check_unique_key(txn)                    # precondition: key unique on left
check_unique_key(idt)                    # precondition: key unique on right
merge(..., validate="one_to_one")        # pandas asserts it too
check_no_row_multiplication(joined, txn) # postcondition: rows unchanged
```

**Result:** 590,540 in → **590,540 out.** Proven, not hoped.

---

## Part 2 — Memory

### The problem, as arithmetic

```
590,540 rows × 394 columns × 8 bytes = 1.86 GB
```

...for the numbers alone. Your machine has 15.7 GB total but only ~1.3 GB
free with Chrome open.

### What a "dtype" is

**Definition:** the storage format for a column — how many bytes each value
takes and what it can represent.

| dtype | Bytes | Holds |
|---|---|---|
| `float64` | 8 | ~16 significant digits (pandas default) |
| `float32` | 4 | ~7 significant digits |
| `int8` | 1 | whole numbers −128 to 127 |
| `category` | ~1 | repeated text stored efficiently |

### Downcasting: float64 → float32

**New term — Downcasting:** storing a number in fewer bytes.

**Tiny example.** Store the number `3.14159265358979`:
- `float64` → keeps all of it (8 bytes)
- `float32` → keeps `3.141593` (4 bytes)

For a *count* of transactions, you don't need 16 digits. Half the memory,
zero loss.

```
Before: 590,540 × 394 × 8 = 1.86 GB
After:  590,540 × 394 × 4 = 0.93 GB    ← what we actually got: 928 MB ✓
```

### ⚠️ The trap: WHEN you downcast matters enormously

```python
# ✗ WRONG — peaks HIGHER than doing nothing
df = pd.read_csv(...)              # 1.86 GB in memory
df = df.astype('float32')          # builds 0.93 GB while 1.86 GB still exists
                                   # PEAK = 2.79 GB  ← worse!

# ✓ RIGHT — the big version never exists
plan = build_dtype_map(sample)     # read 20k rows, learn the types
df = pd.read_csv(..., dtype=plan)  # PEAK = 0.93 GB
```

**This is a great interview answer** because most people know *that* you
downcast but not *that the order matters*.

### Why we did NOT downcast the money column

**New term — Mantissa (significand):** the digits of a floating-point number,
as opposed to its exponent. `float32` has 24 bits of it, giving ~7
significant decimal digits.

**Tiny example.** With float32:
- `31937.39` → stored as about `31937.389` (error ≈ 0.001)

One transaction: fine. But feature engineering will **sum**, **average**, and
take **ratios** of amounts across thousands of rows, and rounding errors
**compound**.

**The saving would have been 2.4 MB out of a gigabyte.** Terrible trade.

**Our final dtype spread proves it worked:**

```
float32   399   ← anonymised features, halved
category   32   ← low-cardinality text
int32       2   ← TransactionID, TransactionDT
int8        1   ← isFraud (only needs 0 or 1)
float64     1   ← TransactionAmt  ★ deliberately NOT downcast
```

That single `float64` is the whole argument in one line.

### `category` dtype

**Tiny example.** `ProductCD` has 590,540 rows but only 5 distinct values
(`W`, `C`, `H`, `R`, `S`).

- As text: 590,540 separate Python string objects (~35 MB)
- As `category`: 5 strings + 590,540 tiny integer codes (~0.6 MB)

**~50× smaller.** But only when values repeat a lot — we skip it for
near-unique columns like `DeviceInfo`, where the dictionary would be as big
as the data.

---

## Part 3 — Parquet

**New term — Columnar storage:** stores data column-by-column instead of
row-by-row.

**Tiny example.**

```
Row-based (CSV):     1,£50,W  |  2,£30,C  |  3,£900,W
Columnar (Parquet):  [1,2,3] | [£50,£30,£900] | [W,C,W]
```

Because a column is all one type, it compresses far better — and you can read
3 columns out of 434 without touching the rest.

**Our real numbers:**

| | CSV | Parquet | Gain |
|---|---|---|---|
| Disk | 678 MB | **76.7 MB** | **8.8× smaller** |
| Load time | 25.5 s | **0.63 s** | **40× faster** |
| Keeps dtypes? | ❌ no | ✅ yes | |

40× matters: later stages reload constantly. 100 reloads = 42 minutes saved.

**Bonus:** Spark reads Parquet natively, so Stage 10's PySpark port is a
*port*, not a rewrite.

---

## Part 4 — The Data Contract

**New term — Data contract:** assertions about your data that the code
*refuses to run without*. Not documentation — executable checks.

Think of it as a type signature for your data.

| Check | Catches | Tiny example of the disaster |
|---|---|---|
| `check_required_columns` | Renamed/dropped column | Source renames `isFraud`→`is_fraud`, everything breaks confusingly |
| `check_unique_key` | Non-unique key | The fan-out disaster above |
| `check_no_row_multiplication` | Fan-out actually happened | 590,540 → 590,547 rows |
| `check_target` | Null or invalid labels | A `NaN` label read as "not fraud" |
| `check_time_column` | Nulls in time | Can't order rows → can't split safely |
| `check_fraud_rate` | Truncated download | Only half the file downloaded; rate is now 1.8% |
| orphan-key check | Wrong file pair | You loaded train identity + test transactions |

### Why crash instead of warn?

**Tiny example.** Your download truncates at 50%. Without the fraud-rate
check:
1. Pipeline runs fine ✓
2. Model trains fine ✓
3. Scores look plausible ✓
4. Three weeks later you notice the fraud rate was 1.8%, not 3.5%
5. **Every experiment since then is void**

With the check: it crashes in 30 seconds and tells you why.

---

## Part 5 — Leakage: what ingestion is FORBIDDEN to do

**New term — Data leakage:** training on information that wouldn't be
available at prediction time. It inflates your offline score and the model
then fails in production.

### The rule

> **Ingestion may RESHAPE data. It may not LEARN from data.**

| Allowed (reshaping) | Forbidden (learning) |
|---|---|
| Join two tables | `fillna(df.mean())` |
| Change a dtype | `StandardScaler().fit(df)` |
| Rename a column | Target encoding |
| Sort by time | `SelectKBest` |

**Why?** Reshaping is per-row — nothing crosses between rows. Learning
aggregates *across* rows, and if those rows include the test period, the
future has entered your training data.

### The classic leak, with a tiny example

```python
df['card1'] = df['card1'].fillna(df['card1'].mean())   # ✗ LEAKED
```

Say your data covers Jan–Dec, and you'll test on December.

That `mean()` includes December's values. Every imputed January row now
carries a whisper of December. Your model "knows" something about the future.

**The fix** — do it *after* the split, fitted on train only:

```python
X_train, X_test = split_by_time(df)
imputer = SimpleImputer(strategy='median').fit(X_train)   # ✓ train only
X_train = imputer.transform(X_train)
X_test  = imputer.transform(X_test)     # transform, never re-fit
```

**Memorise this:** `fit` on train, `transform` on everything.

### ⚠️ The dataset-specific trap: label lag

This is the most sophisticated point in Stage 1.

**How `isFraud` is actually created** (per Vesta's documentation): a fraud
label comes from a **reported chargeback**. Fraud is then propagated forward
to transactions linked by card/email/address. Anything not reported within
**120 days** is labelled legitimate.

**New term — Chargeback:** when a customer disputes a charge and the bank
reverses it. It's how fraud gets *discovered* — often weeks after the event.

**Three consequences:**

**1. Label maturity.** A transaction from last week doesn't have a
trustworthy label yet — the chargeback hasn't arrived. Training on immature
labels means training on systematically mislabelled negatives.

**2. Label noise in the negatives.** Unreported fraud is labelled `0`.

**New term — Positive-Unlabelled (PU) learning:** where your "negative" class
actually contains hidden positives.

**Tiny example.** 100 transactions, 5 are actually fraud, but only 3 were
reported. Your labels say "3 fraud, 97 legit" — but 2 of those 97 are lying.
Your model is penalised for correctly flagging them.

**3. Circularity.** Because fraud propagates forward across linked cards, a
feature like "prior transactions on this card" partly encodes the *labelling
mechanism* itself.

> **Say this in an interview.** It shows you understand the **business
> process that generated the data**, not just the CSV. It's the single
> strongest talking point in this stage.

---

## Part 6 — Reproducibility

**New term — SHA-256:** a cryptographic hash. Change one byte of a file and
you get a completely different 64-character fingerprint.

**Tiny example.**
```
"hello"  → 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
"hellp"  → 7d1a54127b222502f5b79b5fb0803061152a44f92b37e23c6527baf665d4da9a
```
One letter changed → totally different hash.

**Why we use it.** The dataset is 678 MB — far too big for git. So instead of
storing the bytes, we store the **fingerprint**:

```json
"sources": {
  "transaction": { "bytes": 683737808, "sha256": "4a7f2e..." }
}
```

Now any model is traceable to the exact bytes that produced it. If someone
re-runs and gets a different hash, you know immediately that results aren't
comparable.

> **Reproducibility = provenance, not storage.** Commit the fingerprint and
> the download script, not the data.

---

## Our Real Results

| Metric | Result | What it means |
|---|---|---|
| transaction | 590,540 × 394 | matches published |
| identity | 144,233 × 41 | matches published |
| **joined** | **590,540 × 434** | 394+41−1. **Zero rows gained or lost** ✓ |
| **fraud** | **20,663 (3.499%)** | matches published 3.5% ✓ |
| **imbalance** | **1 fraud per 27.6 legit** | why accuracy is useless |
| **identity coverage** | **24.42%** | INNER would have deleted 75.6% |
| time span | **182.0 days** | the budget for the temporal split |
| **memory** | **928 MB** | vs ~1.9 GB float64 — halved ✓ |
| **disk** | 678 MB → **76.7 MB** | 8.8× smaller |
| **reload** | 25.5s → **0.63s** | 40× faster |

**Testing:** 12 tests, **16 seconds, no dataset needed** — they use synthetic
fixtures that reproduce every real failure mode (duplicate keys, fan-out, NaN
labels, hyphenated columns).

---

## Interview Q&A

### Q1. Walk me through your ingestion pipeline.
Eight steps. Sniff dtypes on a 20,000-row sample and build an explicit dtype
plan; full read with that plan so the float64 version never exists; normalise
hyphenated column names; validate each table separately — required columns,
key uniqueness, orphan keys; LEFT join with `validate="one_to_one"`; pin exact
types on key, target, time and amount; validate again post-join — target
domain, time nullability, class balance; sort by time; write Parquet plus a
SHA-256 provenance manifest.

### Q2. Why LEFT and not INNER?
Identity coverage is 24%, so INNER would delete 75.6% of the data — including
75% of my fraud examples, which I can't spare at 3.5% prevalence. It would
also introduce selection bias: transactions that carry device fingerprints
aren't a random sample, so I'd train on one population and deploy on another.
And EDA later confirmed missingness is one of the strongest signals in the
dataset, so INNER would have deleted that too.

### Q3. How do you know the join didn't corrupt the data?
Three layers. Precondition: assert the key is unique on both sides — that's
what makes a 1:1 join possible. During: `validate="one_to_one"` so pandas
raises independently. Postcondition: assert output rows equal input rows.
590,540 in, 590,540 out.

### Q4. How did you fit 394 columns in memory?
Sample-based dtype planning. Read 20,000 rows to learn each column's kind,
build a dtype map, then pass it to `read_csv` so the downcast happens at parse
time. That's ~1.9 GB to 928 MB. Importantly *not* `read_csv` then `.astype` —
that holds both copies at once and peaks higher than doing nothing.

### Q5. Why is `TransactionAmt` float64 when everything else is float32?
It's money. float32 gives ~7 significant digits, so around £31,000 the
resolution is roughly 0.002 — and feature engineering will sum, average and
ratio these, where rounding error compounds. The saving would have been 2.4 MB
out of a gigabyte.

### Q6. What is data leakage and how did you prevent it at this stage?
Training on information unavailable at prediction time. At ingestion the risk
is fitted transformations — imputation, scaling, encoding — each learns a
statistic over the whole timeline including the test period. So my rule is
that ingestion may reshape data but may not learn from it. All fitted steps
live inside an sklearn Pipeline after the temporal split.

### Q7. Why not use the competition's test CSV as your test set?
It has no labels — you can't compute a metric on it. It's for leaderboard
submission. I hold out the final time period of the labelled data instead,
which also better simulates production: train on the past, score the future.

### Q8. What's in your manifest and why?
SHA-256 of each input, row and column shapes, positive count and fraud rate,
imbalance ratio, identity coverage, time range, memory footprint, per-step
timings, and library versions. It makes results traceable — I can prove two
experiments used identical inputs, or prove they didn't. In a bank that's an
audit requirement.

### Q9. Your pipeline crashes if the fraud rate is off by 0.5pp. Isn't that brittle?
Intentional. It catches a truncated download or a wrong file — failures that
otherwise produce a *plausible* dataset and a silently wrong model. I'd rather
have a loud false alarm I can check in 30 seconds than a quiet corruption I
find in Stage 5. And the tolerance is a config value, so widening it is a
deliberate, reviewable change.

### Q10. Why synthetic data in your tests?
Speed — 16 seconds versus a 1.3 GB download. CI — must run without Kaggle
credentials. And precision of intent: I *construct* the failure I want to test.
You can't reliably provoke a duplicated key or a fan-out with real data.

---

## Common Mistakes

1. `read_csv()` then `.astype()` — peaks higher than doing nothing
2. INNER join because NaNs are annoying — biases the sample
3. `fillna(0)` on the target — invents non-fraud labels
4. `fillna(df.mean())` before splitting — the canonical leak
5. `train_test_split(shuffle=True)` on time data — fictional AUC
6. Optimising accuracy — 96.5% free by predicting nothing
7. `pd.concat([train, test])` before feature engineering — textbook leakage
8. Committing the dataset to git — repo becomes unclonable
9. **No row-count check after a join** — the most common silent corruption
10. Unanchored `.gitignore` patterns — `data/` also matches `src/pkg/data/`

---

## Files

| File | Purpose |
|---|---|
| `src/risklens/data/dtypes.py` | Dtype planning, column normalisation, categoricals |
| `src/risklens/data/validate.py` | The 7 contract assertions |
| `src/risklens/data/ingest.py` | 8-step pipeline, manifest, `load_joined()` |
| `scripts/download_data.py` | Kaggle fetch with graceful fallback |
| `scripts/run_ingest.py` | CLI entry point |
| `tests/test_ingest.py` | 12 tests on synthetic fixtures |
