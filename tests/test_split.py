"""Tests for the temporal split - the leakage firewall.

These matter more than they look. A split bug does not crash; it produces a
model with a wonderful score that fails in production. So each test below
encodes one property that must hold for the reported metrics to be honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.data.split import (
    SECONDS_PER_DAY,
    SplitError,
    compute_boundaries,
    make_masks,
    temporal_split,
    validate_masks,
)

CFG = {
    "method": "temporal",
    "train_frac": 0.70,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "embargo_days": 1,
}


def make_df(n: int = 5_000, days: int = 180, fraud_rate: float = 0.035) -> pd.DataFrame:
    """Transactions spread evenly over `days`, with fraud sprinkled throughout.

    Fraud is spread across the whole period on purpose: if it were clustered
    at the start, the test partition would legitimately have no positives and
    we would be testing the wrong failure.
    """
    rng = np.random.default_rng(7)
    t = np.linspace(SECONDS_PER_DAY, SECONDS_PER_DAY + days * SECONDS_PER_DAY, n)
    y = (rng.random(n) < fraud_rate).astype(int)
    return pd.DataFrame(
        {"TransactionDT": t.astype(int), "isFraud": y, "x": rng.normal(size=n)}
    )


# ------------------------------------------------------------- boundaries ---
def test_boundaries_are_strictly_ordered():
    df = make_df()
    b = compute_boundaries(
        df["TransactionDT"], train_frac=0.70, val_frac=0.15, embargo_days=1
    )
    assert b.train_end < b.val_start < b.val_end < b.test_start


def test_embargo_creates_a_real_gap():
    """The gap between train_end and val_start must equal the embargo."""
    df = make_df()
    b = compute_boundaries(
        df["TransactionDT"], train_frac=0.70, val_frac=0.15, embargo_days=2
    )
    assert b.val_start - b.train_end == 2 * SECONDS_PER_DAY
    assert b.test_start - b.val_end == 2 * SECONDS_PER_DAY


def test_oversized_embargo_is_rejected():
    """An embargo wider than the data must fail loudly, not silently collapse."""
    df = make_df(days=10)
    with pytest.raises(SplitError, match="too large"):
        compute_boundaries(
            df["TransactionDT"], train_frac=0.70, val_frac=0.15, embargo_days=30
        )


def test_fractions_must_leave_room_for_test():
    df = make_df()
    with pytest.raises(SplitError, match="must be < 1"):
        compute_boundaries(
            df["TransactionDT"], train_frac=0.80, val_frac=0.25, embargo_days=1
        )


# ------------------------------------------------------------------ masks ---
def test_partitions_never_overlap():
    """The core anti-leakage property: no row in two partitions."""
    df = make_df()
    masks, _, _ = temporal_split(
        df, time_col="TransactionDT", target_col="isFraud", split_cfg=CFG
    )
    stacked = np.vstack([m.to_numpy() for m in masks.values()])
    assert stacked.sum(axis=0).max() <= 1


def test_partitions_are_strictly_chronological():
    """max(train) < min(val) and max(val) < min(test). Catches any shuffle."""
    df = make_df()
    masks, _, _ = temporal_split(
        df, time_col="TransactionDT", target_col="isFraud", split_cfg=CFG
    )
    t = df["TransactionDT"]
    assert t[masks["train"]].max() < t[masks["val"]].min()
    assert t[masks["val"]].max() < t[masks["test"]].min()


def test_embargo_actually_drops_rows():
    """Rows inside an embargo window belong to no partition - by design."""
    df = make_df()
    _, _, summary = temporal_split(
        df, time_col="TransactionDT", target_col="isFraud", split_cfg=CFG
    )
    assert summary["rows_dropped_to_embargo"] > 0
    assert summary["rows_assigned"] < summary["rows_total"]


def test_no_embargo_assigns_every_row():
    """With embargo=0 the partitions should tile the whole period."""
    df = make_df()
    cfg = {**CFG, "embargo_days": 0}
    _, _, summary = temporal_split(
        df, time_col="TransactionDT", target_col="isFraud", split_cfg=cfg
    )
    assert summary["rows_dropped_to_embargo"] == 0


def test_train_is_the_earliest_period():
    """Sanity: we train on the past and test on the future, not the reverse."""
    df = make_df()
    masks, _, _ = temporal_split(
        df, time_col="TransactionDT", target_col="isFraud", split_cfg=CFG
    )
    t = df["TransactionDT"]
    assert t[masks["train"]].mean() < t[masks["test"]].mean()


def test_approximate_partition_sizes():
    """~70/15/15, allowing for embargo losses."""
    df = make_df()
    _, _, summary = temporal_split(
        df, time_col="TransactionDT", target_col="isFraud", split_cfg=CFG
    )
    p = summary["partitions"]
    assert 0.66 < p["train"]["share"] < 0.72
    assert 0.12 < p["val"]["share"] < 0.17
    assert 0.12 < p["test"]["share"] < 0.17


# ------------------------------------------------------------- validation ---
def test_partition_without_positives_is_rejected():
    """A partition with no fraud makes PR-AUC and recall undefined."""
    df = make_df(fraud_rate=0.0)
    df.loc[:10, "isFraud"] = 1  # positives only at the very start
    with pytest.raises(SplitError, match="no fraud cases"):
        temporal_split(
            df, time_col="TransactionDT", target_col="isFraud", split_cfg=CFG
        )


def test_overlapping_masks_are_rejected():
    df = make_df(n=100)
    t = df["TransactionDT"]
    bad = {
        "train": t <= t.quantile(0.7),
        "val": t >= t.quantile(0.5),   # deliberately overlaps train
        "test": t >= t.quantile(0.9),
    }
    with pytest.raises(SplitError, match="more than one partition"):
        validate_masks(bad, df["isFraud"], t)


def test_random_split_method_is_refused():
    """RiskLens must not permit a random split at all."""
    df = make_df()
    with pytest.raises(SplitError, match="only permits 'temporal'"):
        temporal_split(
            df,
            time_col="TransactionDT",
            target_col="isFraud",
            split_cfg={**CFG, "method": "random"},
        )


def test_summary_reports_fraud_rate_per_partition():
    df = make_df()
    _, _, summary = temporal_split(
        df, time_col="TransactionDT", target_col="isFraud", split_cfg=CFG
    )
    for name in ("train", "val", "test"):
        p = summary["partitions"][name]
        assert p["fraud"] > 0
        assert 0 < p["fraud_rate"] < 0.2
        assert p["span_days"] > 0
