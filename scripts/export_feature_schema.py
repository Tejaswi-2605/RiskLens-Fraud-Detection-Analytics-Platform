"""Export the exact training-time feature schema for the serving path.

The bug this fixes
------------------
The API built a feature frame with the same COLUMNS as training but different
DTYPES, and XGBoost rejected it:

    ValueError: DataFrame.dtypes for data must be int, float, bool or category.
    Invalid columns: ProductCD: object, card4: object, ...

At training, `ProductCD` was a pandas `category`, because Stage 1 converted
low-cardinality strings during ingestion. At serving, `pd.DataFrame([payload])`
produces `object`. Same column name, same value, different dtype.

Why this is the important one
-----------------------------
A raised error is the LUCKY outcome here. The dangerous version is silent.

A pandas `category` is stored as an integer CODE plus a categories list.
XGBoost splits on those codes. If serving builds its own category list from
whatever single row arrived, "visa" might be code 0 at serving but code 3 at
training - and the model would confidently apply a split learned for a
different card network. No error, wrong answer.

So it is not enough to cast to `category`. The **exact same categories, in
the same order** must be used. That is what a `CategoricalDtype` carries, and
that is what this script persists.

Output
------
    models/feature_schema.joblib   an ordered {column: dtype} mapping, where
                                   categorical entries carry their full
                                   category list from the TRAINING partition

Usage
-----
    python scripts/export_feature_schema.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from risklens.config import load_data_config  # noqa: E402
from risklens.data.ingest import load_joined  # noqa: E402
from risklens.data.split import temporal_split  # noqa: E402
from risklens.features.build import add_deterministic_features  # noqa: E402
from risklens.features.entity import build_entity_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger("schema")


def main() -> int:
    cfg = load_data_config()
    md = cfg.root / "models"

    for f in ("frequency_encoder.joblib", "feature_names.joblib"):
        if not (md / f).is_file():
            log.error("missing models/%s - run scripts/run_train.py first", f)
            return 1

    encoder = joblib.load(md / "frequency_encoder.joblib")
    features = joblib.load(md / "feature_names.joblib")

    log.info("rebuilding the training frame to capture its dtypes ...")
    df = load_joined(cfg)
    df = add_deterministic_features(df)
    df = build_entity_features(df, time_col=cfg.time_column,
                               amount_col=cfg.amount_column)
    df = encoder.transform(df)

    masks, _, _ = temporal_split(
        df, time_col=cfg.time_column, target_col=cfg.target, split_cfg=cfg.split
    )
    # Categories must come from the TRAINING partition only. Deriving them
    # from the full frame would leak category levels that appear solely in
    # the test period.
    train = df.loc[masks["train"], features]
    del df

    schema: dict[str, object] = {}
    n_cat = 0
    for col in features:
        dt = train[col].dtype
        if isinstance(dt, pd.CategoricalDtype):
            # Store the CategoricalDtype itself - it carries the ordered
            # category list, which is what makes the integer codes stable.
            schema[col] = pd.CategoricalDtype(categories=dt.categories,
                                              ordered=dt.ordered)
            n_cat += 1
        else:
            schema[col] = dt

    out = md / "feature_schema.joblib"
    joblib.dump(schema, out)

    log.info("wrote %s", out.name)
    print(f"\n  {len(schema)} columns")
    print(f"  {n_cat} categorical (with their full training category lists)")
    print(f"  {len(schema) - n_cat} numeric\n")
    print("  Sample categorical levels captured from TRAIN:")
    shown = 0
    for col, dt in schema.items():
        if isinstance(dt, pd.CategoricalDtype) and shown < 4:
            cats = list(dt.categories)[:6]
            more = "" if len(dt.categories) <= 6 else f" ... (+{len(dt.categories)-6})"
            print(f"    {col:<26} {cats}{more}")
            shown += 1
    print("\n  Serving now casts to these EXACT dtypes, so a category maps to")
    print("  the same integer code it had during training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
