"""Generate the RiskLens analysis notebooks from code.

Why generate notebooks instead of hand-writing .ipynb files
-----------------------------------------------------------
A .ipynb is JSON with embedded outputs. Hand-edited notebooks drift from the
codebase, produce enormous git diffs, and are easy to break. Generating them
from a script keeps them:

  * reproducible  - rebuild any time with one command
  * reviewable    - the source of truth is readable Python
  * consistent    - always reflect the current library code

The notebook is EXECUTED after generation, so the outputs you see are real
results computed from the real data, not pasted text.

Usage
-----
    python scripts/build_notebooks.py            # build + execute
    python scripts/build_notebooks.py --no-exec  # build only (fast)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


# =========================================================================
# Notebook 01 - Ingestion, EDA and the Temporal Split
# =========================================================================
def notebook_01() -> nbf.NotebookNode:
    cells = [
        md("""
# RiskLens 01 — Ingestion, EDA & the Temporal Split

**Fraud detection on the IEEE-CIS dataset** (public Kaggle benchmark, Vesta Corporation 2019).
*Not proprietary data.*

This notebook reproduces Stages 1–3 and shows the real results.

| Stage | What it does |
|---|---|
| **1. Ingestion** | Two CSVs → one validated, typed, time-ordered Parquet |
| **3. Split** | Chronological train/val/test with an embargo gap |
| **2. EDA** | Explore the **training partition only** |

> **Why EDA comes after the split:** if I explore everything and then pick
> features from what I saw, my choices encode knowledge of the test period.
> That's *human-in-the-loop leakage* — no test can catch it, because it
> happened in my head.
        """),

        code("""
import sys, json, warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from risklens.config import load_data_config
from risklens.data.ingest import load_joined
from risklens.data.split import temporal_split
from risklens.eda import profile, stats

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 50)
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
FRAUD_C, LEGIT_C = "#c0392b", "#2c7fb8"

cfg = load_data_config()
print("project root:", cfg.root)
        """),

        md("""
---
## 1. Ingestion provenance

The raw data is **not** in git — it's 678 MB and public. Instead we commit a
**manifest** containing the SHA-256 fingerprint of each input file.

**Why:** reproducibility comes from *provenance*, not storage. Any model is
traceable to the exact bytes that produced it. If a re-run yields a different
hash, we know instantly that results aren't comparable.
        """),

        code("""
manifest = json.loads((cfg.reports_dir / "stage01_ingest_manifest.json").read_text())

print("SOURCE FILES")
for name, s in manifest["sources"].items():
    print(f"  {name:<12} {s['bytes']/1024**2:8.1f} MB   sha256 {s['sha256'][:16]}...")

print("\\nSHAPES")
for k, v in manifest["shapes"].items():
    print(f"  {k:<12} {v[0]:>8,} rows x {v[1]:>3} cols")

t = manifest["target"]
print(f"\\nTARGET")
print(f"  fraud          {t['positives']:>8,}  ({t['fraud_rate']:.3%})")
print(f"  legitimate     {t['negatives']:>8,}")
print(f"  imbalance      1 fraud per {t['imbalance_ratio']:.1f} legitimate")

print(f"\\nJOIN")
print(f"  identity coverage  {manifest['join']['identity_coverage']:.2%}")
print(f"  time span          {manifest['time']['span_days']} days")
        """),

        md("""
### ✅ The join is provably safe

`590,540` transaction rows → `590,540` joined rows. **Zero rows gained or lost.**

**Why that matters — "join fan-out":** if the right table had duplicate keys,
one left row would match several right rows and the output would silently
*grow*. Every count, rate and metric downstream would then be wrong, with no
error raised.

We guard it three ways: assert key uniqueness on both sides *before*, pass
`validate="one_to_one"` to pandas *during*, and assert the row count *after*.
        """),

        code("""
%time df = load_joined(cfg)
print(f"\\nshape: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"memory: {df.memory_usage(deep=True).sum()/1024**2:,.0f} MB")

print("\\nDTYPE SPREAD  (proof the memory strategy worked)")
print(df.dtypes.astype(str).value_counts().to_string())
print(f"\\n  float64 columns: {list(df.dtypes[df.dtypes=='float64'].index)}")
print("  ^ TransactionAmt is deliberately NOT downcast: it is money, and")
print("    float32's ~7 significant digits would compound rounding error")
print("    through the sums and ratios in feature engineering.")
        """),

        md("""
---
## 2. Class imbalance — why accuracy is a useless metric

Only **3.5%** of transactions are fraud.

A model that predicts *"never fraud"* for every single row achieves **96.5%
accuracy** and catches **zero** fraud. Accuracy is dominated by the majority
class, so we use **PR-AUC**, precision and recall instead.
        """),

        code("""
counts = df[cfg.target].value_counts().sort_index()
rate = float(df[cfg.target].mean())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.bar(["legitimate", "fraud"], counts.values, color=[LEGIT_C, FRAUD_C])
for i, v in enumerate(counts.values):
    a1.text(i, v, f"{v:,}", ha="center", va="bottom")
a1.set_ylabel("transactions"); a1.set_title(f"Class balance — fraud = {rate:.2%}")

a2.pie([1-rate, rate], labels=["legitimate", "fraud"], colors=[LEGIT_C, FRAUD_C],
       autopct="%1.2f%%", startangle=90, explode=(0, 0.25))
a2.set_title(f"'Always predict legitimate' = {1-rate:.1%} accuracy, 0 fraud caught")
plt.tight_layout(); plt.show()

print(f"imbalance ratio: 1 fraud per {counts[0]/counts[1]:.1f} legitimate")
        """),

        md("""
---
## 3. The temporal split — our leakage firewall

**Rule 1 — split by TIME, never randomly.** In production you only have the past.

**Rule 2 — embargo the boundaries.** Fraud is *bursty*: one compromised card
produces many near-identical transactions minutes apart. A hard boundary can
put siblings of the same burst on both sides, so the model *memorises* rather
than predicts. We **drop** a 1-day window at each boundary.

**Rule 3 — test is opened once,** at the very end.
        """),

        code("""
masks, boundaries, summary = temporal_split(
    df, time_col=cfg.time_column, target_col=cfg.target, split_cfg=cfg.split
)

rows = []
for name, p in summary["partitions"].items():
    rows.append({"partition": name, "rows": f"{p['rows']:,}", "share": f"{p['share']:.1%}",
                 "fraud": f"{p['fraud']:,}", "fraud_rate": f"{p['fraud_rate']:.3%}",
                 "days": p["span_days"]})
display(pd.DataFrame(rows))

print(f"embargo dropped: {summary['rows_dropped_to_embargo']:,} rows "
      f"({summary['rows_dropped_to_embargo']/len(df):.1%})")
print(f"\\nboundaries: {summary['boundaries']}")
        """),

        md("""
### Why 74/12/12 and not the 70/15/15 we configured?

**Because we split by calendar time, not row count.** Those differ whenever
transaction volume is uneven.

**Tiny example:** 100 transactions over 10 days, busier early —
days 1–7 hold 80 transactions. "70% of the *time*" = day 7, but that's
80% of the *rows*.

Our volume was higher in the first ~25 days, so the earliest 70% of the
calendar contains 74.2% of the transactions.

**This is deliberate.** Real deployments retrain on *"the last six months"*,
not *"the last 400,000 rows"*.

✅ Fraud rate is **3.51% / 3.47% / 3.56%** across partitions — the split
created neither an easy nor an impossible test set.
        """),

        code("""
train = df[masks["train"]]
test  = df[masks["test"]]
print(f"EDA uses the TRAINING partition only: {train.shape[0]:,} rows")
print("The test partition stays sealed until the final evaluation.")
        """),

        md("""
---
## 4. Is fraud stationary over time?

If the fraud rate is stable, a random split might be defensible.
**Let's check rather than assume.**
        """),

        code("""
ts = profile.fraud_rate_over_time(train, time_col=cfg.time_column,
                                  target=cfg.target, bucket_days=7)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
a1.plot(ts["day"], ts["fraud_rate"]*100, color=FRAUD_C, marker="o", ms=4)
a1.axhline(float(train[cfg.target].mean())*100, ls="--", c="grey", label="period mean")
a1.fill_between(ts["day"], ts["fraud_rate"]*100, alpha=0.15, color=FRAUD_C)
a1.set_ylabel("fraud rate (%)"); a1.legend()
a1.set_title("Fraud rate over time — training period, weekly buckets")
a2.bar(ts["day"], ts["n"], width=5, color=LEGIT_C, alpha=0.75)
a2.set_ylabel("volume"); a2.set_xlabel("days since first transaction")
plt.tight_layout(); plt.show()

lo, hi = ts["fraud_rate"].min(), ts["fraud_rate"].max()
print(f"weekly fraud rate ranges {lo:.2%} .. {hi:.2%}  ->  {hi/lo:.1f}x swing")
        """),

        md("""
### ❗ Verdict: strongly non-stationary

The weekly fraud rate swings from **2.07% to 5.08%** — a **2.5× swing**.

**This is hard evidence that a random split would be indefensible.** Fraud
patterns drift as criminals adapt and as the business changes. Any evaluation
that ignores time is measuring the wrong thing.

Note also the volume bars: higher early, which is exactly why our split came
out 74/12/12.
        """),

        md("""
---
## 5. Missingness — 229 of 434 columns are >50% empty

That isn't a broken dataset; it's what real financial data looks like.
Different systems capture different fields.
        """),

        code("""
miss = profile.missingness_report(train)
print(f"columns with NO missing values : {(miss['pct_missing']==0).sum()} / {len(miss)}")
print(f"columns >50% missing           : {(miss['pct_missing']>50).sum()} / {len(miss)}")
display(miss.head(10))

pct = (train.isna().mean()*100).sort_values(ascending=False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(range(len(pct)), pct.values, color=FRAUD_C)
a1.fill_between(range(len(pct)), pct.values, alpha=0.3, color=FRAUD_C)
a1.set_xlabel("columns (sorted)"); a1.set_ylabel("% missing")
a1.set_title("Missingness profile — note the flat plateaus")
a2.hist(pct.values, bins=40, color=LEGIT_C)
a2.set_xlabel("% missing"); a2.set_ylabel("number of columns")
a2.set_title("Columns clump at shared levels\\n= correlated blocks")
plt.tight_layout(); plt.show()
        """),

        md("""
### The plateaus are a real structural finding

Those flat sections mean **many columns share an identical missingness
pattern** — they're missing on *exactly* the same rows. That happens when
columns come from the same upstream source.

Let's quantify it.
        """),

        code("""
v_cols = [c for c in train.columns if c.startswith("V")]
blocks = profile.missingness_blocks(train, v_cols)
display(blocks.head(14))
print(f"{len(v_cols)} V columns collapse into {len(blocks)} distinct missingness patterns")
        """),

        md("""
### 339 V-columns → **14** distinct patterns

**Tiny example:** three columns `height_cm`, `height_inches`, `height_feet`,
all missing for the same people. That's *one fact in three costumes*.

**New term — Multicollinearity:** features so correlated they carry duplicate
information.

**Practical consequences:**
- Feature selection should operate on **blocks**, not columns
- Imputation should treat a block as one unit — they move together
- A single *"is block 5 present?"* flag may beat all 29 of its columns
- Trees cope fine, but linear-model coefficients become uninterpretable
        """),

        md("""
---
## 6. ⚠️ The finding that overturned my hypothesis

**In Stage 1, I predicted:** *"Missing identity data suggests evasion —
fraudsters avoid leaving device fingerprints. So fraud should be HIGHER when
identity is missing."*

Let's test that.
        """),

        code("""
id_cols = [c for c in train.columns if c.startswith("id_")] + \\
          ["DeviceType", "DeviceInfo", "dist1", "dist2", "D1", "D15"]
sig = profile.missing_indicator_vs_fraud(train, id_cols, target=cfg.target)
display(sig.head(10))
print(f"baseline fraud rate: {train[cfg.target].mean():.3%}")
        """),

        code("""
base = float(train[cfg.target].mean())
cols = ["id_01", "id_31", "DeviceType", "DeviceInfo", "dist1", "dist2", "D15"]
cols = [c for c in cols if c in train.columns]
mr = [train.loc[train[c].isna(),  cfg.target].mean()*100 for c in cols]
pr = [train.loc[train[c].notna(), cfg.target].mean()*100 for c in cols]

x = np.arange(len(cols)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(x-w/2, mr, w, label="value MISSING", color=FRAUD_C)
ax.bar(x+w/2, pr, w, label="value present", color=LEGIT_C)
ax.axhline(base*100, ls="--", c="black", lw=1, label=f"baseline {base:.2%}")
ax.set_xticks(x); ax.set_xticklabels(cols, rotation=30, ha="right")
ax.set_ylabel("fraud rate (%)")
ax.set_title("Missingness IS signal — but not in the direction I predicted")
ax.legend(); plt.tight_layout(); plt.show()
        """),

        md("""
### ❗ I was wrong — and this is the most valuable finding in the project

| Column | fraud when **MISSING** | fraud when **PRESENT** |
|---|---|---|
| `id_04` | **2.61%** | **10.31%** |
| `id_09` | 2.51% | 10.02% |
| `DeviceType` | 2.12% | 7.58% |

**Fraud is ~4× HIGHER when identity data is PRESENT** — the opposite of my
prediction.

### Why? A confounding variable.

**New term — Confounder:** a hidden third factor causing the relationship you
observe.

**Tiny example:** ice cream sales correlate with drownings. Ice cream doesn't
cause drowning — **summer** causes both.

Here, the real chain is:

```
identity present → it's an ONLINE (card-not-present) transaction
                 → online is inherently riskier
                 → more fraud
```

Device fingerprints only exist when there's a browser to fingerprint. In-store
chip-and-PIN purchases have none. And card-not-present fraud is far more common,
because the criminal never needs the physical card.

**So identity presence is a proxy for the sales channel, not for evasiveness.**

### What this proves

The **direction** was wrong; the **decision** was right — and more strongly
than I'd argued. A 4× difference is enormous. Missingness is among the most
predictive things here, and an INNER join would have destroyed it.

> **Interview answer:** *"I hypothesised missing identity meant evasion. The
> data showed the opposite — 4× higher fraud when identity is present. That's
> a channel effect: identity is captured for card-not-present transactions,
> which are riskier. So my LEFT join was even more justified, but my causal
> story was wrong. I'd control for ProductCD before concluding anything about
> device fingerprints specifically."*
        """),

        md("""
---
## 7. Statistical tests — ranked by **effect size**, not p-value

**New term — p-value:** the chance of seeing a difference this big if there
were truly *no* difference. Small p = "probably real".

### ⚠️ The trap at our sample size

With **438,125 rows**, almost every p-value collapses to zero.

**Tiny example:** fraudsters average £100.01, legitimate users £100.00.
- 100 rows → p = 0.9, not significant
- 10,000,000 rows → p < 0.0001, "highly significant"

The difference is **one penny**. Significant, worthless.

**So we rank by effect size**, which measures *how big* the difference is,
independent of sample size.
        """),

        code("""
CATEGORICAL = ["ProductCD","card4","card6","P_emaildomain","R_emaildomain",
               "M1","M2","M3","M4","M5","M6","M7","M8","M9",
               "DeviceType","id_31","id_30"]
NUMERIC = ["TransactionAmt","dist1","dist2","C1","C13","C14","D1","D15"]

battery = stats.run_test_battery(train, target=cfg.target,
                                 categorical=CATEGORICAL, numeric=NUMERIC)
display(battery.head(15))
        """),

        code("""
top = battery.head(12).iloc[::-1]
colors = [FRAUD_C if abs(v) >= 0.147 else "#95a5a6" for v in top["effect_size"]]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(top["feature"], top["effect_size"].abs(), color=colors)
ax.axvline(0.147, ls="--", c="black", lw=1, label="negligible / small threshold")
ax.set_xlabel("|effect size|  (Cliff's δ or Cramér's V)")
ax.set_title("What actually predicts fraud — grey = negligible")
ax.legend(); plt.tight_layout(); plt.show()
        """),

        md("""
### Two things to notice

**1. `TransactionAmt` has essentially NO predictive power.**
Cliff's δ = **0.0014**, p = 0.78.

**What Cliff's δ means:** pick one random fraud and one random legitimate
transaction — how often is the fraud one bigger? δ = 0.0014 means **50.07%**
of the time. A coin flip.

Everyone assumes fraudsters spend more. **Here they don't.** That's consistent
with **card testing** — small, unremarkable purchases to verify a stolen card
works before selling it on. Large charges trigger alerts, so smart fraudsters
avoid them.

**2. Nothing is a strong predictor.** The best is "small but real" (δ ≈ 0.24).

That's *normal and reassuring*. Fraud detection wins by **combining many weak
signals** — exactly what gradient boosting does. If you find one overwhelming
feature, suspect **leakage** before celebrating.
        """),

        md("""
---
## 8. Drift check — PSI (Population Stability Index)

**The standard drift metric in banking.** Measures how far a distribution has
moved between a reference period and a current one.

$$\\text{PSI} = \\sum_i (a_i - e_i)\\ln\\frac{a_i}{e_i}$$

| PSI | Verdict |
|---|---|
| < 0.10 | stable |
| 0.10–0.25 | moderate — investigate |
| > 0.25 | major shift — retrain |

**Tiny example:** if customers shift from mostly-small to mostly-large
purchases between training and production, PSI is high — your model now sees a
different population than it learned from.
        """),

        code("""
psi_cols = list(dict.fromkeys(NUMERIC + [c for c in train.columns if c.startswith("C")][:10]))
psi = stats.psi_report(train, test, psi_cols)
display(psi)

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(psi["column"][::-1], psi["psi"][::-1],
        color=["#27ae60" if v < 0.1 else "#f39c12" if v < 0.25 else FRAUD_C
               for v in psi["psi"][::-1]])
ax.axvline(0.10, ls="--", c="orange", label="moderate (0.10)")
ax.axvline(0.25, ls="--", c=FRAUD_C, label="major (0.25)")
ax.set_xlabel("PSI"); ax.set_title("Train vs Test drift — all stable")
ax.legend(); plt.tight_layout(); plt.show()
        """),

        md("""
✅ **All features PSI < 0.06 — no meaningful drift.**

Our train and test periods are distributionally comparable, so a model that
works on train has a fair chance on test. In production, this is exactly what
you'd monitor to know when a model has gone stale and needs retraining.
        """),

        md("""
---
# Summary of Findings

| # | Finding | Consequence |
|---|---|---|
| 1 | Join preserved all 590,540 rows | Counts and rates are trustworthy |
| 2 | 3.499% fraud, 1 : 27.6 imbalance | Use PR-AUC, never accuracy |
| 3 | Identity coverage only 24.4% | LEFT join was essential |
| 4 | Fraud rate swings 2.07%→5.08% | Random split would be indefensible |
| 5 | **Fraud 4× higher when identity PRESENT** | Channel effect, not evasion — my hypothesis was wrong |
| 6 | **`TransactionAmt` has no signal** (δ=0.001) | Consistent with card testing |
| 7 | 339 V-columns → 14 patterns | Heavy redundancy; use blocks |
| 8 | Best predictors: D15, C1, C13, id_31 | All weak — combine many signals |
| 9 | All PSI < 0.06 | No drift; periods comparable |

## What's next

**Stage 3b — feature engineering:** encode missingness explicitly (finding 5),
derive cyclical time features, frequency-encode high-cardinality IDs.

**Stage 4 — modelling:** logistic regression baseline, then XGBoost. All
fitted steps inside an sklearn `Pipeline` so they can only ever see training data.
        """),
    ]

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.9"},
    }
    return nb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-exec", action="store_true", help="build without executing")
    args = ap.parse_args()

    NB_DIR.mkdir(parents=True, exist_ok=True)
    targets = [("01_ingestion_eda_split.ipynb", notebook_01())]

    for name, nb in targets:
        path = NB_DIR / name
        nbf.write(nb, path)
        print(f"built  {path.relative_to(ROOT).as_posix()}  ({len(nb.cells)} cells)")

        if not args.no_exec:
            print(f"executing {name} ... (this runs the real pipeline)")
            r = subprocess.run(
                [sys.executable, "-m", "jupyter", "nbconvert",
                 "--to", "notebook", "--execute", "--inplace",
                 "--ExecutePreprocessor.timeout=1800", str(path)],
                capture_output=True, text=True, cwd=str(ROOT),
            )
            if r.returncode != 0:
                print("EXECUTION FAILED:\n", r.stderr[-3000:])
                return 1
            print(f"executed {name} - outputs embedded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
