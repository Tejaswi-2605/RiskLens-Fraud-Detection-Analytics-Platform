"""Stage 2 - data profiling.

IMPORTANT: every function here is meant to run on the TRAINING PARTITION ONLY.

Why that rule exists
--------------------
If you explore the whole dataset and then choose features based on what you
saw, YOU are the leak. Your feature choices encode knowledge of the test
period, and no code can detect it afterwards. This is "human-in-the-loop
leakage" and it is the form most people never think about.

So the temporal split (Stage 3) is defined FIRST, by a rule, and EDA sees
only the training period. The test partition stays sealed until the very end.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


# --------------------------------------------------------------- missing ---
def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness, dtype and cardinality, worst first.

    Missingness is the single most informative thing about this dataset:
    the identity block and many V columns are largely absent, and *why*
    they are absent is itself a modelling question (MCAR vs MNAR).
    """
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "pct_missing": round(100.0 * n_missing / n, 2) if n else 0.0,
                "n_missing": n_missing,
                "n_unique": int(s.nunique(dropna=True)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["pct_missing", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def missingness_blocks(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Group columns that share an IDENTICAL missingness pattern.

    Why this matters for IEEE-CIS
    -----------------------------
    The 339 anonymised V columns were not engineered independently. Large
    groups of them are missing on exactly the same rows, which means they
    came from the same upstream source or the same feature-generation batch.

    Consequences:
      * They are highly redundant - a block of 40 columns may carry roughly
        one column's worth of independent information.
      * Imputing them independently is wrong; the block is the unit.
      * A single "is this block present" indicator often beats the columns.

    Implementation: hash each column's null-mask; identical hashes share a
    pattern. Hashing beats pairwise comparison - O(n_cols) not O(n_cols^2).
    """
    present = [c for c in columns if c in df.columns]
    sig: dict[str, list[str]] = {}
    for col in present:
        key = pd.util.hash_pandas_object(df[col].isna(), index=False).sum()
        sig.setdefault(str(key), []).append(col)

    rows = [
        {
            "block_id": i,
            "n_columns": len(cols),
            "pct_missing": round(100.0 * df[cols[0]].isna().mean(), 2),
            "columns": ", ".join(cols[:6]) + (" ..." if len(cols) > 6 else ""),
        }
        for i, cols in enumerate(
            sorted(sig.values(), key=len, reverse=True), start=1
        )
    ]
    return pd.DataFrame(rows)


# -------------------------------------------------------------- temporal ---
def fraud_rate_over_time(
    df: pd.DataFrame, *, time_col: str, target: str, bucket_days: int = 1
) -> pd.DataFrame:
    """Fraud rate per time bucket.

    This is the single most important EDA output in the project: it is the
    EVIDENCE for the temporal split. If the fraud rate is visibly unstable
    over time, a random split is indefensible and you can say so with a chart
    rather than an opinion.
    """
    t0 = int(df[time_col].min())
    bucket = ((df[time_col] - t0) // (bucket_days * SECONDS_PER_DAY)).astype(int)
    out = (
        df.groupby(bucket)[target]
        .agg(n="size", fraud="sum")
        .assign(fraud_rate=lambda d: d["fraud"] / d["n"])
        .reset_index(names="bucket")
    )
    out["day"] = out["bucket"] * bucket_days
    return out


def hour_of_day_profile(
    df: pd.DataFrame, *, time_col: str, target: str
) -> pd.DataFrame:
    """Fraud rate by hour of day.

    TransactionDT is seconds from an unknown origin, so we cannot recover a
    calendar date - but modulo arithmetic still gives a valid *relative*
    hour-of-day. The origin offset is unknown but CONSTANT, so the shape of
    the daily cycle is real even though the absolute clock time is not.
    """
    hour = ((df[time_col] // 3600) % 24).astype(int)
    return (
        df.groupby(hour)[target]
        .agg(n="size", fraud="sum")
        .assign(fraud_rate=lambda d: d["fraud"] / d["n"])
        .reset_index(names="hour")
    )


# ----------------------------------------------------------- categorical ---
def categorical_fraud_rates(
    df: pd.DataFrame, column: str, *, target: str, min_count: int = 100, top: int = 15
) -> pd.DataFrame:
    """Fraud rate per category, with a Wilson-style lower bound on volume.

    `min_count` guards against the classic beginner error: a category with
    3 transactions and 1 fraud shows a 33% fraud rate and looks like the most
    predictive signal in the dataset. It is noise. Rare categories need a
    volume floor before their rate means anything.
    """
    g = (
        df.groupby(column, observed=True)[target]
        .agg(n="size", fraud="sum")
        .assign(fraud_rate=lambda d: d["fraud"] / d["n"])
        .reset_index()
    )
    baseline = float(df[target].mean())
    g["lift"] = g["fraud_rate"] / baseline if baseline else np.nan
    return (
        g[g["n"] >= min_count]
        .sort_values("fraud_rate", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


# -------------------------------------------------- missingness as signal ---
def missing_indicator_vs_fraud(
    df: pd.DataFrame, columns: list[str], *, target: str
) -> pd.DataFrame:
    """Does the ABSENCE of a column predict fraud?

    This directly tests the claim that justified the LEFT join in Stage 1:
    that missing identity data is informative rather than merely inconvenient.

    For each column we compare the fraud rate where it is missing against the
    fraud rate where it is present. A large gap means "is_missing" deserves to
    be an explicit feature, and it means dropping those rows would have
    destroyed real signal.
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        miss = df[col].isna()
        if miss.all() or not miss.any():
            continue
        r_missing = float(df.loc[miss, target].mean())
        r_present = float(df.loc[~miss, target].mean())
        rows.append(
            {
                "column": col,
                "pct_missing": round(100.0 * miss.mean(), 2),
                "fraud_rate_when_missing": round(r_missing, 5),
                "fraud_rate_when_present": round(r_present, 5),
                "abs_gap": round(abs(r_missing - r_present), 5),
                "ratio": round(r_missing / r_present, 3) if r_present else np.nan,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_gap", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------- numeric ---
def numeric_summary_by_class(
    df: pd.DataFrame, column: str, *, target: str
) -> pd.DataFrame:
    """Distribution of a numeric column, split by class.

    Reported on the log scale for money: TransactionAmt is heavily
    right-skewed, so the mean is dragged by a few large values and the median
    is the more honest centre.
    """
    out = (
        df.groupby(target)[column]
        .agg(
            n="size",
            missing=lambda s: int(s.isna().sum()),
            mean="mean",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            p99=lambda s: s.quantile(0.99),
            max="max",
        )
        .reset_index()
    )
    return out
