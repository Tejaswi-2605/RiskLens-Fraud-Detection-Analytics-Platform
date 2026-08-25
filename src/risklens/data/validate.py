"""Data contract checks for Stage 1.

Philosophy
----------
An ingestion pipeline that "works" but silently produces 1.4M rows instead of
590k, or a target column containing a stray 2, is worse than one that crashes:
the bug survives into a model and shows up months later as a bad risk decision.

So ingestion asserts a *contract* and raises on violation. Each check below
maps to a specific, real failure mode - they are not decoration.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class DataContractError(ValueError):
    """Raised when the ingested data violates an assumption we rely on."""


def check_required_columns(df: pd.DataFrame, required: list[str], *, name: str) -> None:
    """Fail if the source schema changed under us (renamed/dropped column)."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataContractError(f"{name}: missing required column(s) {missing}")


def check_unique_key(df: pd.DataFrame, key: str, *, name: str) -> None:
    """The join key must identify a row uniquely.

    If it does not, a left join *multiplies* rows (fan-out) and every
    downstream count, rate and class balance becomes wrong.
    """
    if key not in df.columns:
        raise DataContractError(f"{name}: join key '{key}' not present")
    n_dupes = int(df[key].duplicated().sum())
    if n_dupes:
        raise DataContractError(
            f"{name}: join key '{key}' is not unique ({n_dupes} duplicate values)"
        )


def check_no_row_multiplication(
    joined: pd.DataFrame, left: pd.DataFrame, *, name: str = "join"
) -> None:
    """A LEFT join must preserve the left table's row count exactly."""
    if len(joined) != len(left):
        raise DataContractError(
            f"{name}: row count changed during join "
            f"({len(left)} -> {len(joined)}). The right table's key is not unique."
        )


def check_target(df: pd.DataFrame, target: str, allowed: list[int]) -> None:
    """Binary target must be exactly {0, 1} and fully populated.

    A missing label is not a 'legitimate transaction' - treating NaN as 0 is a
    classic way to manufacture a fake fraud rate.
    """
    if target not in df.columns:
        raise DataContractError(f"target column '{target}' not present")
    n_null = int(df[target].isna().sum())
    if n_null:
        raise DataContractError(f"target '{target}' has {n_null} missing labels")
    observed = set(pd.unique(df[target]).tolist())
    unexpected = observed - set(allowed)
    if unexpected:
        raise DataContractError(
            f"target '{target}' contains unexpected values {sorted(unexpected)}; "
            f"expected {allowed}"
        )


def check_time_column(df: pd.DataFrame, time_col: str) -> None:
    """Time must be present, non-null and non-decreasing-capable.

    Stage 7 splits the data by time. If TransactionDT has nulls we cannot
    order rows, and a leakage-free temporal split becomes impossible.
    """
    if time_col not in df.columns:
        raise DataContractError(f"time column '{time_col}' not present")
    n_null = int(df[time_col].isna().sum())
    if n_null:
        raise DataContractError(f"time column '{time_col}' has {n_null} nulls")
    if (df[time_col] < 0).any():
        raise DataContractError(f"time column '{time_col}' contains negative offsets")


def check_fraud_rate(rate: float, expected: float, tolerance: float) -> None:
    """Sanity-anchor the class balance against the published dataset.

    Catches the 'I accidentally loaded the wrong file / only a chunk' bug.
    """
    if abs(rate - expected) > tolerance:
        raise DataContractError(
            f"fraud rate {rate:.4%} is outside expected "
            f"{expected:.4%} +/- {tolerance:.4%}. Wrong file or partial read?"
        )
