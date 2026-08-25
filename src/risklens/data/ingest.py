"""Stage 1 - Data Ingestion.

Contract of this module
-----------------------
IN : data/raw/train_transaction.csv, data/raw/train_identity.csv  (immutable)
OUT: data/interim/transactions_joined.parquet                     (typed, joined)
     reports/stage01_ingest_manifest.json                         (provenance)

What ingestion is explicitly NOT allowed to do
----------------------------------------------
No imputation, no scaling, no encoding, no outlier removal, no feature
engineering, no dropping of "useless" columns. Every one of those is a
*modelling decision* that must be fitted on the training split only. Doing it
here - before the temporal split in Stage 7 - would fit it on future data and
leak. Ingestion only: read, rename, type, join, verify, persist.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risklens.config import DataConfig, load_data_config
from risklens.data import validate as V
from risklens.data.dtypes import (
    build_dtype_map,
    memory_mb,
    normalise_columns,
    to_categorical,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
@dataclass
class IngestManifest:
    """Everything needed to prove *which* data produced a given model.

    Serialised to reports/ and committed to git. The parquet itself is
    gitignored (too big), so this manifest is the reproducibility anchor: if
    the sha256 of a raw CSV ever differs from what is recorded here, results
    are not comparable and we find out immediately instead of never.
    """

    created_utc: str
    python: str
    pandas: str
    numpy: str
    platform: str
    sources: dict[str, Any] = field(default_factory=dict)
    shapes: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    time: dict[str, Any] = field(default_factory=dict)
    join: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    timings_sec: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 so we never load a 650 MB file into RAM just to hash it."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_table(csv_path: Path, *, keep_exact: set[str], label: str) -> pd.DataFrame:
    """Read one CSV with an explicit, memory-efficient dtype plan."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{label} file not found: {csv_path}\n"
            f"Run:  python scripts/download_data.py"
        )

    t0 = time.perf_counter()
    dtype_plan = build_dtype_map(csv_path, keep_exact=keep_exact)
    df = pd.read_csv(csv_path, dtype=dtype_plan)

    # Canonicalise `id-01` -> `id_01` immediately, before anything else can
    # come to depend on the raw spelling.
    renames = normalise_columns(list(df.columns))
    if renames:
        log.info("%s: normalising %d hyphenated column names", label, len(renames))
        df = df.rename(columns=renames)

    df = to_categorical(df)
    log.info(
        "%s loaded: shape=%s memory=%.1f MB in %.1fs",
        label, df.shape, memory_mb(df), time.perf_counter() - t0,
    )
    return df


def join_identity(
    transaction: pd.DataFrame, identity: pd.DataFrame, *, key: str
) -> pd.DataFrame:
    """LEFT join identity onto transaction.

    Why LEFT and not INNER
    ----------------------
    Identity data (device, browser, network) exists for only a minority of
    transactions. An INNER join would throw away most of the dataset AND
    silently change the population we model - the subset that has identity
    data is not a random sample of all transactions.

    Crucially, *the absence of identity data is itself predictive signal*. We
    keep those rows with NaNs so later stages can encode "identity missing"
    as a feature rather than deleting the evidence.

    `validate="one_to_one"` makes pandas itself assert that neither side's key
    repeats - a belt-and-braces guard against join fan-out.
    """
    joined = transaction.merge(identity, on=key, how="left", validate="one_to_one")
    V.check_no_row_multiplication(joined, transaction, name="transaction<-identity")
    return joined


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def ingest(
    cfg: DataConfig | None = None, *, write: bool = True
) -> tuple[pd.DataFrame, IngestManifest]:
    """Run the full Stage 1 pipeline: load -> validate -> join -> verify -> persist."""
    cfg = cfg or load_data_config()
    timings: dict[str, float] = {}

    keep_exact = {cfg.join_key, cfg.target, cfg.time_column, cfg.amount_column}

    # ---- 1. Load ----------------------------------------------------------
    t0 = time.perf_counter()
    txn = load_table(cfg.transaction_csv, keep_exact=keep_exact, label="transaction")
    idt = load_table(cfg.identity_csv, keep_exact=keep_exact, label="identity")
    timings["load"] = time.perf_counter() - t0

    # ---- 2. Per-table contract checks (BEFORE joining) --------------------
    V.check_required_columns(
        txn, cfg.contract["required_transaction_columns"], name="transaction"
    )
    V.check_unique_key(txn, cfg.join_key, name="transaction")
    V.check_unique_key(idt, cfg.join_key, name="identity")

    # Identity rows must refer to real transactions. Orphans mean we have been
    # handed a mismatched file pair (e.g. train identity + test transactions).
    orphans = int((~idt[cfg.join_key].isin(txn[cfg.join_key])).sum()) if len(idt) else 0
    if orphans:
        raise V.DataContractError(
            f"{orphans} identity rows reference a {cfg.join_key} absent from the "
            "transaction table. Mismatched file pair?"
        )

    # ---- 3. Join ----------------------------------------------------------
    t0 = time.perf_counter()
    df = join_identity(txn, idt, key=cfg.join_key)
    timings["join"] = time.perf_counter() - t0

    # ---- 4. Pin exact dtypes on the four columns that carry meaning -------
    df[cfg.join_key] = df[cfg.join_key].astype("int32")
    df[cfg.target] = df[cfg.target].astype("int8")
    df[cfg.time_column] = df[cfg.time_column].astype("int32")
    df[cfg.amount_column] = df[cfg.amount_column].astype("float64")  # money: no downcast

    # ---- 5. Post-join contract checks -------------------------------------
    V.check_target(df, cfg.target, cfg.contract["target_values"])
    V.check_time_column(df, cfg.time_column)

    fraud_rate = float(df[cfg.target].mean())
    V.check_fraud_rate(
        fraud_rate,
        cfg.expected["approx_fraud_rate"],
        cfg.expected["fraud_rate_tolerance"],
    )

    # ---- 6. Canonical ordering by time ------------------------------------
    # Not a modelling decision - just a deterministic order so that Stage 7's
    # temporal split and any future windowed feature are reproducible.
    # mergesort = stable, so ties keep their original relative order.
    df = df.sort_values(cfg.time_column, kind="mergesort", ignore_index=True)

    # ---- 7. Manifest ------------------------------------------------------
    matched = df[cfg.join_key].isin(idt[cfg.join_key])
    dt = df[cfg.time_column]

    manifest = IngestManifest(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python=sys.version.split()[0],
        pandas=pd.__version__,
        numpy=np.__version__,
        platform=platform.platform(),
        sources={
            "transaction": {
                "path": cfg.transaction_csv.relative_to(cfg.root).as_posix(),
                "bytes": cfg.transaction_csv.stat().st_size,
                "sha256": sha256_file(cfg.transaction_csv),
            },
            "identity": {
                "path": cfg.identity_csv.relative_to(cfg.root).as_posix(),
                "bytes": cfg.identity_csv.stat().st_size,
                "sha256": sha256_file(cfg.identity_csv),
            },
        },
        shapes={
            "transaction": list(txn.shape),
            "identity": list(idt.shape),
            "joined": list(df.shape),
        },
        target={
            "column": cfg.target,
            "positives": int(df[cfg.target].sum()),
            "negatives": int((df[cfg.target] == 0).sum()),
            "fraud_rate": fraud_rate,
            "imbalance_ratio": round(
                float((df[cfg.target] == 0).sum() / max(int(df[cfg.target].sum()), 1)), 2
            ),
        },
        time={
            "column": cfg.time_column,
            "min": int(dt.min()),
            "max": int(dt.max()),
            "span_days": round(float((int(dt.max()) - int(dt.min())) / 86400), 2),
        },
        join={
            "type": "left",
            "key": cfg.join_key,
            "identity_rows_matched": int(matched.sum()),
            "identity_coverage": round(float(matched.mean()), 4),
        },
        memory={"joined_mb": round(memory_mb(df), 1)},
        timings_sec={k: round(v, 2) for k, v in timings.items()},
    )

    # ---- 8. Persist -------------------------------------------------------
    if write:
        cfg.interim_dir.mkdir(parents=True, exist_ok=True)
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        df.to_parquet(
            cfg.joined_parquet, engine="pyarrow", compression="snappy", index=False
        )
        manifest.timings_sec["write_parquet"] = round(time.perf_counter() - t0, 2)
        manifest.memory["parquet_mb"] = round(
            cfg.joined_parquet.stat().st_size / 1024**2, 1
        )

        with open(cfg.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(manifest), fh, indent=2)
        log.info("wrote %s", cfg.joined_parquet)
        log.info("wrote %s", cfg.manifest_path)

    return df, manifest


def load_joined(cfg: DataConfig | None = None) -> pd.DataFrame:
    """Fast reload of the Stage 1 output, for use by every later stage.

    Later stages must call THIS, never re-parse the CSVs:
    one ingestion, one artefact, one source of truth.
    """
    cfg = cfg or load_data_config()
    if not cfg.joined_parquet.is_file():
        raise FileNotFoundError(
            f"{cfg.joined_parquet} not found. Run:  python scripts/run_ingest.py"
        )
    return pd.read_parquet(cfg.joined_parquet, engine="pyarrow")
