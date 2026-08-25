"""Stage 2 entry point - EDA on the TRAINING PARTITION ONLY.

The order of operations here is the whole point:

    1. load the joined data
    2. apply the temporal split          <-- FIRST
    3. explore ONLY the training rows    <-- never test

If we explored everything and then picked features from what we saw, our
choices would encode knowledge of the test period. That is human-in-the-loop
leakage, and no assertion can catch it after the fact. So the split comes
first, by rule, and the test partition stays sealed.

Usage
-----
    python scripts/run_eda.py
    python scripts/run_eda.py --no-plots        # tables only, faster
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from risklens.config import load_data_config  # noqa: E402
from risklens.data.ingest import load_joined  # noqa: E402
from risklens.data.split import temporal_split  # noqa: E402
from risklens.eda import profile, stats  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage02")

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

# Named categorical features in this dataset (the rest are anonymised).
CATEGORICAL = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "id_31", "id_30",
]
# Interpretable numerics - the ones we can actually reason about.
NUMERIC = ["TransactionAmt", "dist1", "dist2", "C1", "C13", "C14", "D1", "D15"]


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_data_config()
    out_dir = cfg.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading joined parquet ...")
    df = load_joined(cfg)
    log.info("loaded %s", df.shape)

    # ---- STEP 1: split BEFORE looking at anything ------------------------
    masks, boundaries, split_summary = temporal_split(
        df,
        time_col=cfg.time_column,
        target_col=cfg.target,
        split_cfg=cfg.split,
    )
    with open(out_dir / "stage03_split_summary.json", "w", encoding="utf-8") as fh:
        json.dump(split_summary, fh, indent=2)

    train = df[masks["train"]]
    test = df[masks["test"]]          # used ONLY for drift measurement, never explored
    log.info("EDA will use the TRAINING partition only: %s", train.shape)

    findings: dict[str, object] = {"split": split_summary}

    # ---- STEP 2: missingness --------------------------------------------
    section("MISSINGNESS (training partition)")
    miss = profile.missingness_report(train)
    miss.to_csv(out_dir / "stage02_missingness.csv", index=False)
    print(miss.head(15).to_string(index=False))
    fully_present = int((miss["pct_missing"] == 0).sum())
    over_half = int((miss["pct_missing"] > 50).sum())
    print(f"\n  columns with NO missing values : {fully_present} / {len(miss)}")
    print(f"  columns >50% missing           : {over_half} / {len(miss)}")
    findings["missingness"] = {
        "columns_total": len(miss),
        "columns_fully_present": fully_present,
        "columns_over_50pct_missing": over_half,
        "worst": miss.head(5).to_dict("records"),
    }

    # ---- STEP 3: correlated missingness blocks in the V columns ---------
    section("CORRELATED MISSINGNESS - the V block structure")
    v_cols = [c for c in train.columns if c.startswith("V")]
    blocks = profile.missingness_blocks(train, v_cols)
    blocks.to_csv(out_dir / "stage02_v_blocks.csv", index=False)
    print(blocks.head(12).to_string(index=False))
    print(f"\n  {len(v_cols)} V columns collapse into {len(blocks)} distinct")
    print("  missingness patterns -> heavy redundancy, and a strong argument")
    print("  for treating each BLOCK as one unit rather than 339 features.")
    findings["v_blocks"] = {"n_v_columns": len(v_cols), "n_blocks": len(blocks)}

    # ---- STEP 4: does MISSING predict fraud? ----------------------------
    section("IS MISSINGNESS ITSELF A SIGNAL?  (tests the LEFT-join decision)")
    id_cols = [c for c in train.columns if c.startswith("id_")] + [
        "DeviceType", "DeviceInfo", "dist1", "dist2", "D1", "D15",
    ]
    sig = profile.missing_indicator_vs_fraud(train, id_cols, target=cfg.target)
    sig.to_csv(out_dir / "stage02_missing_as_signal.csv", index=False)
    print(sig.head(12).to_string(index=False))
    baseline = float(train[cfg.target].mean())
    print(f"\n  baseline fraud rate = {baseline:.3%}")
    print("  A large gap between 'missing' and 'present' means the ABSENCE of")
    print("  this field predicts fraud - so dropping those rows (INNER join)")
    print("  would have deleted real signal.")
    findings["missing_as_signal"] = sig.head(10).to_dict("records")

    # ---- STEP 5: fraud rate over time -----------------------------------
    section("TEMPORAL STABILITY - the evidence for a time-based split")
    ts = profile.fraud_rate_over_time(
        train, time_col=cfg.time_column, target=cfg.target, bucket_days=7
    )
    ts.to_csv(out_dir / "stage02_fraud_over_time.csv", index=False)
    print(ts[["day", "n", "fraud", "fraud_rate"]].to_string(index=False))
    rmin, rmax = float(ts["fraud_rate"].min()), float(ts["fraud_rate"].max())
    print(f"\n  weekly fraud rate ranges {rmin:.3%} .. {rmax:.3%}"
          f"  ({rmax / rmin:.1f}x swing)")
    print("  Non-stationary -> a RANDOM split would be indefensible.")
    findings["temporal"] = {
        "weekly_fraud_rate_min": round(rmin, 5),
        "weekly_fraud_rate_max": round(rmax, 5),
        "swing_ratio": round(rmax / rmin, 2),
    }

    # ---- STEP 6: categorical fraud rates --------------------------------
    section("CATEGORICAL FRAUD RATES (lift vs baseline)")
    cat_tables = {}
    for col in ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain"]:
        if col not in train.columns:
            continue
        t = profile.categorical_fraud_rates(
            train, col, target=cfg.target, min_count=500, top=8
        )
        cat_tables[col] = t
        print(f"\n--- {col} ---")
        print(t.to_string(index=False))
    findings["categoricals"] = {
        k: v.to_dict("records") for k, v in cat_tables.items()
    }

    # ---- STEP 7: statistical tests --------------------------------------
    section("STATISTICAL TESTS (ranked by EFFECT SIZE, not p-value)")
    battery = stats.run_test_battery(
        train, target=cfg.target, categorical=CATEGORICAL, numeric=NUMERIC
    )
    battery.to_csv(out_dir / "stage02_stat_tests.csv", index=False)
    print(battery.to_string(index=False))
    print("\n  At n=413k almost everything is 'significant'. The effect size")
    print("  column is what actually ranks these features.")
    findings["stat_tests"] = battery.head(10).to_dict("records")

    # ---- STEP 8: drift between train and test (PSI) ---------------------
    section("DRIFT: train vs test partition (Population Stability Index)")
    # dict.fromkeys preserves order while de-duplicating: C1 appears in both
    # NUMERIC and the C-prefix sweep, and would otherwise be reported twice.
    psi_cols = list(dict.fromkeys(
        NUMERIC + [c for c in train.columns if c.startswith("C")][:10]
    ))
    psi = stats.psi_report(train, test, psi_cols)
    psi.to_csv(out_dir / "stage02_psi.csv", index=False)
    print(psi.to_string(index=False))
    print("\n  PSI < 0.10 stable | 0.10-0.25 moderate | > 0.25 major shift.")
    print("  High-PSI features are the ones most likely to degrade in")
    print("  production and the first things a bank would monitor.")
    findings["psi"] = psi.to_dict("records")

    # ---- persist --------------------------------------------------------
    with open(out_dir / "stage02_eda_findings.json", "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2, default=str)

    if not args.no_plots:
        from risklens.eda import plots
        plots.make_all(train, cfg, ts=ts, out_dir=out_dir / "figures")

    section("DONE")
    print(f"  tables + findings -> {out_dir.relative_to(cfg.root).as_posix()}/")
    if not args.no_plots:
        print(f"  figures           -> {(out_dir / 'figures').relative_to(cfg.root).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
