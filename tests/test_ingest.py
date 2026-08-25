"""Tests for Stage 1 ingestion.

Design note - why these tests use synthetic data
------------------------------------------------
A test suite that requires a 1.3 GB download is a test suite nobody runs.
These tests build tiny CSVs in a temp directory that reproduce the *shape* of
the real problem: a wide numeric table, a partially-matching identity table,
hyphenated column names, NaNs, and an imbalanced binary target.

That lets the whole contract be verified in under a second, in CI, on a
machine that has never seen Kaggle. The real data is then exercised once, by
`scripts/run_ingest.py`, which re-applies the exact same checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.config import DataConfig
from risklens.data.dtypes import build_dtype_map, normalise_columns, to_categorical
from risklens.data.ingest import ingest, join_identity, load_joined
from risklens.data.validate import (
    DataContractError,
    check_no_row_multiplication,
    check_target,
    check_unique_key,
)

N_ROWS = 200
N_FRAUD = 7  # 3.5% - matches the real dataset's class balance


def _make_transaction(n: int = N_ROWS, n_fraud: int = N_FRAUD) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[rng.choice(n, size=n_fraud, replace=False)] = 1
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1000, 1000 + n),
            "isFraud": is_fraud,
            # deliberately unsorted, to prove ingestion sorts by time
            "TransactionDT": rng.permutation(np.arange(86400, 86400 + n * 60, 60)),
            "TransactionAmt": np.round(rng.uniform(1, 5000, n), 2),
            "ProductCD": rng.choice(list("WCHRS"), n),          # low cardinality
            "card1": rng.integers(1000, 20000, n).astype(float),
            "V1": rng.normal(size=n),
            "V2": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
        }
    )


def _make_identity(txn: pd.DataFrame, coverage: float = 0.25) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    k = int(len(txn) * coverage)
    ids = rng.choice(txn["TransactionID"].to_numpy(), size=k, replace=False)
    return pd.DataFrame(
        {
            "TransactionID": np.sort(ids),
            # hyphenated on purpose: mirrors the real test_identity.csv quirk
            "id-01": rng.normal(size=k),
            "id-31": rng.choice(["chrome", "safari", "ie"], k),
            "DeviceType": rng.choice(["desktop", "mobile"], k),
        }
    )


@pytest.fixture()
def cfg(tmp_path):
    """A DataConfig rooted in tmp_path, backed by tiny synthetic CSVs."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    txn, idt = _make_transaction(), None
    idt = _make_identity(txn)
    txn.to_csv(raw_dir / "train_transaction.csv", index=False)
    idt.to_csv(raw_dir / "train_identity.csv", index=False)
    (tmp_path / "pyproject.toml").write_text("")

    raw = {
        "paths": {
            "raw": "data/raw",
            "interim": "data/interim",
            "processed": "data/processed",
            "reports": "reports",
        },
        "files": {
            "transaction": "train_transaction.csv",
            "identity": "train_identity.csv",
        },
        "outputs": {
            "joined": "transactions_joined.parquet",
            "manifest": "stage01_ingest_manifest.json",
        },
        "schema": {
            "join_key": "TransactionID",
            "target": "isFraud",
            "time_column": "TransactionDT",
            "amount_column": "TransactionAmt",
        },
        "contract": {
            "required_transaction_columns": [
                "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
            ],
            "target_values": [0, 1],
            "join_type": "left",
            "allow_row_multiplication": False,
        },
        "expected": {
            "approx_fraud_rate": N_FRAUD / N_ROWS,
            "fraud_rate_tolerance": 0.005,
        },
    }
    return DataConfig(raw=raw, root=tmp_path)


# ---------------------------------------------------------------- dtypes ---
def test_hyphenated_columns_are_normalised():
    assert normalise_columns(["id-01", "id_02", "DeviceType"]) == {"id-01": "id_01"}


def test_dtype_plan_downcasts_features_but_not_money(cfg):
    keep = {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt"}
    plan = build_dtype_map(cfg.transaction_csv, keep_exact=keep)
    assert plan["V1"] == "float32"
    assert plan["card1"] == "float32"
    for protected in keep:
        assert protected not in plan, f"{protected} must not be downcast"


def test_low_cardinality_objects_become_category():
    df = pd.DataFrame({"a": ["x", "y"] * 50, "b": [f"u{i}" for i in range(100)]})
    out = to_categorical(df, max_cardinality_ratio=0.5)
    assert out["a"].dtype == "category"   # 2 distinct / 100 rows -> worth it
    assert out["b"].dtype == object       # 100 distinct / 100 rows -> not


# ------------------------------------------------------------ validation ---
def test_duplicate_join_key_is_rejected():
    df = pd.DataFrame({"TransactionID": [1, 2, 2]})
    with pytest.raises(DataContractError, match="not unique"):
        check_unique_key(df, "TransactionID", name="identity")


def test_row_multiplication_is_rejected():
    left = pd.DataFrame({"k": [1, 2]})
    joined = pd.DataFrame({"k": [1, 2, 2]})
    with pytest.raises(DataContractError, match="row count changed"):
        check_no_row_multiplication(joined, left)


def test_missing_label_is_rejected():
    df = pd.DataFrame({"isFraud": [0, 1, np.nan]})
    with pytest.raises(DataContractError, match="missing labels"):
        check_target(df, "isFraud", [0, 1])


def test_unexpected_label_value_is_rejected():
    df = pd.DataFrame({"isFraud": [0, 1, 2]})
    with pytest.raises(DataContractError, match="unexpected values"):
        check_target(df, "isFraud", [0, 1])


def test_fanout_join_raises(cfg):
    """A non-unique right key must fail loudly, not silently duplicate rows."""
    txn = pd.DataFrame({"TransactionID": [1, 2], "x": [1.0, 2.0]})
    idt = pd.DataFrame({"TransactionID": [1, 1], "y": [9.0, 8.0]})
    with pytest.raises((DataContractError, pd.errors.MergeError)):
        join_identity(txn, idt, key="TransactionID")


# ------------------------------------------------------- full pipeline ----
def test_ingest_end_to_end(cfg):
    df, manifest = ingest(cfg, write=True)

    # LEFT join semantics: every transaction survives, none is duplicated.
    assert len(df) == N_ROWS
    assert df["TransactionID"].is_unique

    # Identity is optional -> partial coverage, NaNs preserved (not dropped).
    assert 0 < manifest.join["identity_coverage"] < 1
    assert df["id_01"].isna().any()
    assert "id-01" not in df.columns  # normalised at the door

    # Types are pinned where it matters.
    assert df["TransactionID"].dtype == "int32"
    assert df["isFraud"].dtype == "int8"
    assert df["TransactionDT"].dtype == "int32"
    assert df["TransactionAmt"].dtype == "float64"
    assert df["V1"].dtype == "float32"

    # Canonical time ordering, ready for the Stage 7 temporal split.
    assert df["TransactionDT"].is_monotonic_increasing

    # Target is intact.
    assert manifest.target["positives"] == N_FRAUD
    assert manifest.target["fraud_rate"] == pytest.approx(N_FRAUD / N_ROWS)

    # Artefacts + provenance written.
    assert cfg.joined_parquet.is_file()
    assert cfg.manifest_path.is_file()
    assert len(manifest.sources["transaction"]["sha256"]) == 64


def test_ingest_is_deterministic(cfg):
    """Same input -> byte-identical logical output. No hidden randomness."""
    a, _ = ingest(cfg, write=False)
    b, _ = ingest(cfg, write=False)
    pd.testing.assert_frame_equal(a, b)


def test_load_joined_roundtrips(cfg):
    original, _ = ingest(cfg, write=True)
    reloaded = load_joined(cfg)
    assert reloaded.shape == original.shape
    assert reloaded["isFraud"].sum() == original["isFraud"].sum()


def test_missing_raw_file_gives_actionable_error(cfg):
    cfg.transaction_csv.unlink()
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        ingest(cfg, write=False)


# ------------------------------------------------- serving-time robustness ---
def test_frequency_encoder_tolerates_absent_columns():
    """Training/serving skew regression test.

    At training time every source column exists. A live API payload is sparse
    and may omit `id_31`, `DeviceInfo` and others entirely. An earlier version
    indexed the column directly and raised KeyError on the first real request.

    The encoder must still EMIT the `_freq` column, filled with unseen_value.
    Dropping it would change the feature vector's shape and silently shift
    every downstream column - and XGBoost matches positionally as well as by
    name, so that is a confident wrong answer rather than an error.
    """
    from risklens.features.build import FrequencyEncoder

    train = pd.DataFrame({
        "card1": [1, 1, 1, 2],
        "id_31": ["chrome", "chrome", "safari", "ie"],
        "other": [0.1, 0.2, 0.3, 0.4],
    })
    fe = FrequencyEncoder(columns=["card1", "id_31"]).fit(train)

    # a serving payload with card1 present but id_31 entirely absent
    payload = pd.DataFrame({"card1": [1], "other": [0.5]})
    out = fe.transform(payload)

    assert "card1_freq" in out.columns
    assert "id_31_freq" in out.columns, "absent column must still emit its feature"
    assert out["card1_freq"].iloc[0] == pytest.approx(0.75)   # 3 of 4 train rows
    assert out["id_31_freq"].iloc[0] == 0.0                   # never observed
    assert out["id_31_freq"].dtype == "float32"


def test_frequency_encoder_maps_unseen_categories_to_zero():
    """A category present at serving but never seen in training is the rarest case."""
    from risklens.features.build import FrequencyEncoder

    train = pd.DataFrame({"card1": [1, 1, 2]})
    fe = FrequencyEncoder(columns=["card1"]).fit(train)
    out = fe.transform(pd.DataFrame({"card1": [999]}))
    assert out["card1_freq"].iloc[0] == 0.0
