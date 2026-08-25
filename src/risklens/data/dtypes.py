"""Memory-efficient dtype planning for wide CSV files.

The problem
-----------
`train_transaction.csv` is ~590k rows x 394 columns. pandas defaults every
numeric column to float64 (8 bytes). That is roughly:

    590_540 * 394 * 8 bytes  ~=  1.86 GB

...for the numbers alone, before object columns and before the join. On a
normal laptop that is enough to swap or die.

The fix
-------
Almost every numeric column here is a count, a day-delta, or an anonymised
`V` feature. Those do not need 15-17 significant decimal digits. float32
(4 bytes) holds ~7 significant digits and halves the footprint to ~930 MB.

Two columns are deliberately NOT downcast:
  * the join key   -> exact integer identity matters
  * the money column -> never silently lose precision on currency

Strategy: sniff a small sample to learn each column's *kind*, build an
explicit dtype map, then do the full read with that map. This means we never
materialise the float64 version at all - the saving is at read time, not a
post-hoc `astype` (which would need both copies in memory simultaneously).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SNIFF_ROWS = 20_000


def normalise_columns(columns: list[str]) -> dict[str, str]:
    """Map raw CSV headers to canonical names.

    The IEEE-CIS release is inconsistent: the identity columns are called
    `id_01..id_38` in the train file but `id-01..id-38` (hyphens) in the test
    file. If we ignore that, a train-fitted model explodes at scoring time on
    'unknown column'. We normalise hyphens to underscores at the door so the
    rest of the codebase only ever sees one spelling.
    """
    return {c: c.replace("-", "_") for c in columns if "-" in c}


def read_header(csv_path: Path) -> list[str]:
    """Column names only - reads a single row, not the file."""
    return list(pd.read_csv(csv_path, nrows=0).columns)


def build_dtype_map(
    csv_path: Path,
    *,
    keep_exact: set[str],
    sniff_rows: int = SNIFF_ROWS,
) -> dict[str, str]:
    """Infer a memory-efficient dtype for every column from a sample.

    Parameters
    ----------
    keep_exact
        Columns that must NOT be downcast (join key, money, target, time).
    """
    sample = pd.read_csv(csv_path, nrows=sniff_rows)
    plan: dict[str, str] = {}

    for col, dtype in sample.dtypes.items():
        canonical = col.replace("-", "_")
        if canonical in keep_exact:
            continue  # let pandas infer; we fix these explicitly after load
        if pd.api.types.is_numeric_dtype(dtype):
            # float32 not int32: nearly every column here contains NaN, and
            # pandas cannot represent NaN in a numpy integer column.
            plan[col] = "float32"
        # object/string columns are left alone here and converted to
        # `category` AFTER the load (see to_categorical), because pandas
        # cannot build a stable category set while streaming a CSV.

    log.info(
        "dtype plan for %s: %d/%d columns downcast to float32",
        csv_path.name,
        len(plan),
        len(sample.columns),
    )
    return plan


def to_categorical(df: pd.DataFrame, max_cardinality_ratio: float = 0.5) -> pd.DataFrame:
    """Convert low-cardinality object columns to pandas `category`.

    A `category` column stores small integer codes plus one copy of each
    distinct string, instead of one Python string object per row. For a column
    like ProductCD (5 distinct values over 590k rows) that is a ~50x saving.

    We skip high-cardinality columns (e.g. a near-unique device string): there
    the dictionary is as big as the data and `category` only adds overhead.
    """
    out = df
    n = len(df)
    for col in df.columns:
        if df[col].dtype != object:
            continue
        nunique = df[col].nunique(dropna=True)
        if n > 0 and nunique / n <= max_cardinality_ratio:
            out[col] = df[col].astype("category")
    return out


def memory_mb(df: pd.DataFrame) -> float:
    """Deep memory footprint in MB (deep=True actually walks Python strings)."""
    return float(df.memory_usage(deep=True).sum()) / 1024**2
