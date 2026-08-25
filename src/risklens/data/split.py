"""Stage 3 (part 1) - the temporal split. This is the leakage firewall.

Why this file is the most important one in the project
------------------------------------------------------
Every metric RiskLens reports is only as honest as this split. Get it wrong
and every downstream number is fiction - a beautiful PR-AUC that evaporates
in production.

Three rules, and the reasoning behind each:

1. SPLIT BY TIME, NEVER RANDOMLY.
   `train_test_split(shuffle=True)` trains on December to predict June. In
   production you only ever have the past. A random split measures a task you
   will never actually face.

2. EMBARGO THE BOUNDARIES.
   Fraud is bursty: one compromised card produces many transactions minutes
   apart. A hard boundary can drop siblings of the same burst on both sides,
   so the model "predicts" test rows it has effectively already memorised.
   We DROP a window of rows at each boundary rather than assign them.
   (This mirrors purging/embargo from Lopez de Prado's financial ML work.)

3. THE TEST SET IS TOUCHED ONCE.
   Validation drives every decision - features, hyperparameters, threshold.
   Test is opened at the very end, once, to report a number. Peeking at test
   and then changing anything makes YOU the leak, and no code can catch it.

What this module deliberately does NOT do
-----------------------------------------
No fitting. It returns boolean masks over a time column. Every fitted
transformation (imputation, scaling, encoding) happens AFTER this, on the
training partition only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


class SplitError(ValueError):
    """Raised when the requested split is impossible or unsafe."""


@dataclass(frozen=True)
class SplitBoundaries:
    """The cut points, in raw time-column units (seconds for TransactionDT)."""

    train_end: int
    val_start: int
    val_end: int
    test_start: int
    embargo_seconds: int

    def describe(self) -> str:
        return (
            f"train <= {self.train_end:,} | "
            f"val [{self.val_start:,}, {self.val_end:,}] | "
            f"test >= {self.test_start:,} | "
            f"embargo {self.embargo_seconds / SECONDS_PER_DAY:.1f}d"
        )


def compute_boundaries(
    time_values: pd.Series,
    *,
    train_frac: float,
    val_frac: float,
    embargo_days: float,
) -> SplitBoundaries:
    """Find the cut points by TIME QUANTILE, not by row position.

    Quantile-of-time, not quantile-of-rows: transaction volume is uneven
    (weekends, holidays), so "the 70th percentile row" and "70% of the way
    through the period" are different points. We want a clean calendar
    boundary, which is what a real deployment would use.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise SplitError("train_frac and val_frac must each be in (0, 1)")
    if train_frac + val_frac >= 1:
        raise SplitError(
            f"train_frac + val_frac must be < 1 (got {train_frac + val_frac})"
        )

    t_min, t_max = int(time_values.min()), int(time_values.max())
    span = t_max - t_min
    if span <= 0:
        raise SplitError("time column has zero span - cannot split temporally")

    embargo = int(embargo_days * SECONDS_PER_DAY)

    train_end = t_min + int(span * train_frac)
    val_start = train_end + embargo
    val_end = t_min + int(span * (train_frac + val_frac))
    test_start = val_end + embargo

    # `<=` on the embargo edges so that embargo_days=0 is legal: with a zero
    # embargo the boundaries touch but do not overlap, because the mask lower
    # bounds are EXCLUSIVE (see make_masks).
    if not (train_end <= val_start < val_end <= test_start < t_max):
        raise SplitError(
            f"embargo of {embargo_days} day(s) is too large for a "
            f"{span / SECONDS_PER_DAY:.1f}-day span - partitions would collapse"
        )

    return SplitBoundaries(
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        embargo_seconds=embargo,
    )


def make_masks(
    time_values: pd.Series, b: SplitBoundaries
) -> dict[str, pd.Series]:
    """Boolean masks for each partition.

    Rows falling inside an embargo window match NO mask - they are dropped.
    That is intentional and is why the three masks need not cover every row.

    Interval convention: upper bounds are INCLUSIVE, lower bounds EXCLUSIVE.
    So train is (-inf, train_end], val is (val_start, val_end], test is
    (test_start, +inf). That asymmetry is what makes embargo_days=0 safe - a
    row sitting exactly on a boundary lands in the earlier partition only,
    never in both.
    """
    return {
        "train": time_values <= b.train_end,
        "val": (time_values > b.val_start) & (time_values <= b.val_end),
        "test": time_values > b.test_start,
    }


def validate_masks(
    masks: dict[str, pd.Series], y: pd.Series, time_values: pd.Series
) -> None:
    """Assert the split is actually safe. Each check maps to a real failure."""

    # 1. No row may belong to two partitions (would double-count and leak).
    stacked = np.vstack([m.to_numpy() for m in masks.values()])
    overlaps = int((stacked.sum(axis=0) > 1).sum())
    if overlaps:
        raise SplitError(f"{overlaps} rows assigned to more than one partition")

    # 2. Every partition must be non-empty.
    for name, m in masks.items():
        if not m.any():
            raise SplitError(f"partition '{name}' is empty")

    # 3. Every partition must contain positives, or its metrics are undefined.
    #    With ~3.5% prevalence a small partition can genuinely have none.
    for name, m in masks.items():
        n_pos = int(y[m].sum())
        if n_pos == 0:
            raise SplitError(
                f"partition '{name}' contains no fraud cases - "
                "PR-AUC and recall are undefined. Widen the partition."
            )

    # 4. Strict chronological ordering: max(train) < min(val) < max(val) < min(test).
    #    This is the check that would catch an accidental shuffle.
    tr, va, te = (time_values[masks[k]] for k in ("train", "val", "test"))
    if not (tr.max() < va.min() and va.max() < te.min()):
        raise SplitError(
            "partitions are not strictly chronological - "
            "train/val/test overlap in time"
        )


def temporal_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    split_cfg: dict[str, Any],
) -> tuple[dict[str, pd.Series], SplitBoundaries, dict[str, Any]]:
    """Produce validated train/val/test masks over `df`.

    Returns masks (not copies of the data) so callers decide when to
    materialise partitions - important when the frame is ~1 GB.
    """
    if split_cfg.get("method") != "temporal":
        raise SplitError(
            f"unsupported split method {split_cfg.get('method')!r}; "
            "RiskLens only permits 'temporal'"
        )

    t = df[time_col]
    b = compute_boundaries(
        t,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        embargo_days=split_cfg["embargo_days"],
    )
    masks = make_masks(t, b)
    validate_masks(masks, df[target_col], t)

    n = len(df)
    assigned = int(sum(int(m.sum()) for m in masks.values()))
    summary: dict[str, Any] = {
        "boundaries": b.describe(),
        "embargo_days": split_cfg["embargo_days"],
        "rows_total": n,
        "rows_assigned": assigned,
        "rows_dropped_to_embargo": n - assigned,
        "partitions": {},
    }
    for name, m in masks.items():
        sub_y = df.loc[m, target_col]
        sub_t = df.loc[m, time_col]
        summary["partitions"][name] = {
            "rows": int(m.sum()),
            "share": round(float(m.mean()), 4),
            "fraud": int(sub_y.sum()),
            "fraud_rate": round(float(sub_y.mean()), 5),
            "t_min": int(sub_t.min()),
            "t_max": int(sub_t.max()),
            "span_days": round(float((sub_t.max() - sub_t.min()) / SECONDS_PER_DAY), 1),
        }

    log.info("temporal split: %s", b.describe())
    for name, p in summary["partitions"].items():
        log.info(
            "  %-5s %8d rows (%.1f%%)  fraud %5d (%.3f%%)  %.1f days",
            name, p["rows"], p["share"] * 100, p["fraud"],
            p["fraud_rate"] * 100, p["span_days"],
        )
    return masks, b, summary
