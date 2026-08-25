"""Tests for entity-linkage features.

The property that matters
-------------------------
Every aggregate must be BACKWARD-ONLY. If a transaction can see its own value,
or any later transaction, the feature leaks and will look like signal while
generalising to nothing.

That is a subtle bug: it produces better validation scores, not worse ones, so
nothing alerts you. It is exactly the kind of thing that must be a test rather
than a careful comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.features.entity import (
    SECONDS_PER_DAY,
    add_entity_keys,
    add_expanding_entity_features,
    add_normalised_d_columns,
    build_entity_features,
)


def make_df() -> pd.DataFrame:
    """Two entities with known, hand-checkable histories.

    card1=1 / addr1=10 / D1n=0  -> four transactions: 100, 200, 300, 1000
    card1=2 / addr1=20 / D1n=0  -> two transactions:   50, 60
    """
    return pd.DataFrame({
        "TransactionDT": [
            SECONDS_PER_DAY + 0,
            SECONDS_PER_DAY + 100,
            SECONDS_PER_DAY + 200,
            SECONDS_PER_DAY + 300,
            SECONDS_PER_DAY + 400,
            SECONDS_PER_DAY + 500,
        ],
        "TransactionAmt": [100.0, 200.0, 300.0, 50.0, 1000.0, 60.0],
        "card1": [1.0, 1.0, 1.0, 2.0, 1.0, 2.0],
        "addr1": [10.0, 10.0, 10.0, 20.0, 10.0, 20.0],
        "card2": [np.nan] * 6,
        "card3": [np.nan] * 6,
        "D1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })


# ------------------------------------------------------ D normalisation ---
def test_normalised_d_is_stable_across_time_for_one_entity():
    """The whole point: raw D drifts, normalised D does not.

    Two transactions from one card, ten days apart, with D1 growing to match.
    Raw D1 says 30 then 40 - two different situations. Normalised says 70
    both times, which is the fact that links them.
    """
    df = pd.DataFrame({
        "TransactionDT": [100 * SECONDS_PER_DAY, 110 * SECONDS_PER_DAY],
        "D1": [30.0, 40.0],
    })
    out = add_normalised_d_columns(df)
    assert out["D1n"].iloc[0] == out["D1n"].iloc[1] == 70.0


def test_normalisation_creates_one_column_per_d_column():
    df = make_df()
    out = add_normalised_d_columns(df)
    assert "D1n" in out.columns
    assert "day" in out.columns
    # only D1 exists in the fixture, so only D1n should appear
    assert [c for c in out.columns if c.endswith("n") and c[0] == "D"] == ["D1n"]


# ------------------------------------------------------------ uid build ---
def test_uid_groups_the_same_entity_and_separates_different_ones():
    out = add_entity_keys(add_normalised_d_columns(make_df()))
    uid = out["uid"].astype(str)
    assert uid.iloc[0] == uid.iloc[1] == uid.iloc[2] == uid.iloc[4]  # card 1
    assert uid.iloc[3] == uid.iloc[5]                                # card 2
    assert uid.iloc[0] != uid.iloc[3]
    assert out["uid"].nunique() == 2


def test_missing_fields_become_a_group_rather_than_being_dropped():
    """NaN billing region is a consistent, informative group - not a hole."""
    df = make_df()
    df.loc[0, "addr1"] = np.nan
    out = add_entity_keys(add_normalised_d_columns(df))
    assert out["uid"].notna().all()
    assert "na" in str(out["uid"].iloc[0])


# ------------------------------------------- THE CAUSALITY GUARANTEE ------
def test_first_transaction_of_an_entity_has_no_history():
    """Nothing came before it, so every prior-statistic must be undefined."""
    out = build_entity_features(make_df())
    assert out["uid_count_prior"].iloc[0] == 0
    assert pd.isna(out["uid_amt_mean_prior"].iloc[0])
    assert pd.isna(out["uid_amt_ratio"].iloc[0])
    assert pd.isna(out["uid_secs_since_last"].iloc[0])


def test_running_mean_excludes_the_current_row():
    """The .shift(1) that makes this exclusive is the critical line.

    Entity card1=1 sees 100, 200, 300, 1000 in that order:
        row 0  no prior          -> NaN
        row 1  prior {100}       -> 100
        row 2  prior {100,200}   -> 150
        row 4  prior {100,200,300} -> 200
    If the current row leaked in, row 1 would be 150 rather than 100.
    """
    out = build_entity_features(make_df())
    m = out["uid_amt_mean_prior"]
    assert pd.isna(m.iloc[0])
    assert m.iloc[1] == pytest.approx(100.0)
    assert m.iloc[2] == pytest.approx(150.0)
    assert m.iloc[4] == pytest.approx(200.0)


def test_count_prior_counts_only_earlier_transactions():
    out = build_entity_features(make_df())
    assert list(out["uid_count_prior"]) == [0, 1, 2, 0, 3, 1]


def test_amount_ratio_flags_spend_unusual_for_that_entity():
    """1000 against a running mean of 200 is a 5x deviation FOR THIS CLIENT.

    Stage 2 found raw TransactionAmt has no predictive power at all. This
    feature is why: 1000 is unremarkable globally and extraordinary here.
    """
    out = build_entity_features(make_df())
    assert out["uid_amt_ratio"].iloc[4] == pytest.approx(1000.0 / 200.0)


def test_seconds_since_last_measures_velocity():
    """Gaps are to the entity's OWN previous transaction, not the previous row.

    Entity card1=1 transacts at t = 0, 100, 200, 400 (rows 0, 1, 2, 4).
    Rows 3 and 5 belong to card1=2 and must not affect these gaps - which is
    the whole point of grouping before differencing.
    """
    out = build_entity_features(make_df())
    assert pd.isna(out["uid_secs_since_last"].iloc[0])   # no history
    assert out["uid_secs_since_last"].iloc[1] == pytest.approx(100.0)   # 100-0
    assert out["uid_secs_since_last"].iloc[2] == pytest.approx(100.0)   # 200-100
    assert out["uid_secs_since_last"].iloc[4] == pytest.approx(200.0)   # 400-200

    # the other entity is tracked independently: t = 300, 500
    assert pd.isna(out["uid_secs_since_last"].iloc[3])
    assert out["uid_secs_since_last"].iloc[5] == pytest.approx(200.0)


def test_no_feature_can_see_the_future():
    """The strongest statement of the guarantee.

    Append a huge later transaction to one entity. No feature on any EARLIER
    row may change. If one does, that row was reading the future.
    """
    df = make_df()
    base = build_entity_features(df)

    extended = pd.concat([df, pd.DataFrame({
        "TransactionDT": [SECONDS_PER_DAY + 9_000],
        "TransactionAmt": [999_999.0],
        "card1": [1.0], "addr1": [10.0],
        "card2": [np.nan], "card3": [np.nan], "D1": [1.0],
    })], ignore_index=True)
    after = build_entity_features(extended)

    cols = [c for c in base.columns if c.startswith(("uid_", "card1_"))]
    for c in cols:
        pd.testing.assert_series_equal(
            base[c], after[c].iloc[: len(base)], check_names=False,
            obj=f"{c} changed when a LATER transaction was added",
        )


def test_row_order_is_restored():
    """Aggregation sorts internally; callers must get their order back."""
    df = make_df()
    out = build_entity_features(df)
    assert list(out.index) == list(df.index)
    pd.testing.assert_series_equal(out["TransactionAmt"], df["TransactionAmt"])
