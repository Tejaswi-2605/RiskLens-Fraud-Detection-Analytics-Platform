"""Stages 3b-5 entry point: features -> models -> evaluation -> risk engine.

Pipeline
--------
    1. load the Stage 1 Parquet
    2. add DETERMINISTIC features (row-wise, safe anywhere)
    3. temporal split                      <-- the firewall
    4. fit FrequencyEncoder on TRAIN ONLY  <-- the only cross-row learning
    5. train Logistic Regression baseline
    6. train XGBoost with early stopping on validation
    7. evaluate both: PR-AUC, operating points, cost-optimal threshold
    8. persist models + metrics

Usage
-----
    python scripts/run_train.py
    python scripts/run_train.py --sample 150000    # faster iteration
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from risklens.config import load_data_config  # noqa: E402
from risklens.data.ingest import load_joined  # noqa: E402
from risklens.data.split import temporal_split  # noqa: E402
from risklens.features.build import (  # noqa: E402
    FREQ_ENCODE_COLS,
    FrequencyEncoder,
    add_deterministic_features,
    select_model_features,
)
from risklens.models.evaluate import CostModel, evaluate_full  # noqa: E402
from risklens.models.train import (  # noqa: E402
    build_baseline_pipeline,
    build_xgboost,
    scale_pos_weight,
    split_column_types,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage04")

BASELINE_SAMPLE = 120_000  # LogReg on 438k x one-hot is slow and adds nothing


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n  {t}\n{'=' * 74}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=0,
                    help="use only the most recent N rows (faster iteration)")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    cfg = load_data_config()
    models_dir = cfg.root / "models"
    models_dir.mkdir(exist_ok=True)
    t_start = time.perf_counter()

    # ---- 1. load ---------------------------------------------------------
    log.info("loading parquet ...")
    df = load_joined(cfg)
    if args.sample:
        df = df.tail(args.sample).reset_index(drop=True)
        log.info("SAMPLED to most recent %s rows", f"{len(df):,}")
    log.info("loaded %s", df.shape)

    # ---- 2. deterministic features (safe before the split) ---------------
    section("STAGE 3b - DETERMINISTIC FEATURES")
    t0 = time.perf_counter()
    df = add_deterministic_features(df)
    print(f"  built in {time.perf_counter() - t0:.1f}s -> {df.shape[1]} columns")
    print("  These are ROW-WISE: computable for a single transaction at the")
    print("  API with no dataset access. That is what makes them safe to")
    print("  compute before the split - nothing is learned across rows.")

    # ---- 3. temporal split ----------------------------------------------
    section("STAGE 3 - TEMPORAL SPLIT")
    masks, _, split_summary = temporal_split(
        df, time_col=cfg.time_column, target_col=cfg.target, split_cfg=cfg.split
    )
    train_idx, val_idx = masks["train"], masks["val"]

    y_train = df.loc[train_idx, cfg.target].to_numpy()
    y_val = df.loc[val_idx, cfg.target].to_numpy()
    amt_val = df.loc[val_idx, cfg.amount_column].to_numpy()

    # ---- 4. frequency encoding: FIT ON TRAIN ONLY ------------------------
    section("STAGE 3b - FREQUENCY ENCODING (fitted on TRAIN only)")
    fe = FrequencyEncoder(columns=FREQ_ENCODE_COLS)
    fe.fit(df[train_idx])                      # <-- fit sees TRAIN only
    df = fe.transform(df)                      # <-- transform applies everywhere
    print(f"  encoded {len(fe.columns_)} high-cardinality columns")
    for c in fe.columns_[:5]:
        print(f"    {c:<16} {len(fe.freq_maps_[c]):>7,} distinct values -> 1 numeric column")
    print("\n  Counts come from TRAIN ONLY. Using train+test would leak: a")
    print("  card's future frequency would inform its past prediction.")

    features = select_model_features(df, target=cfg.target)
    spec = split_column_types(df, features)
    print(f"\n  model features: {len(features)} "
          f"({len(spec.numeric)} numeric, {len(spec.categorical)} categorical)")
    print("  EXCLUDED: TransactionID and TransactionDT - both increase with")
    print("  time, so a tree could use them to identify WHICH PERIOD a row is")
    print("  from. That is leakage through the index.")

    X_train = df.loc[train_idx, features]
    X_val = df.loc[val_idx, features]
    del df
    gc.collect()

    results: dict[str, object] = {"split": split_summary,
                                 "n_features": len(features)}
    cost_model = CostModel()

    # ---- 5. baseline: logistic regression --------------------------------
    if not args.skip_baseline:
        section("STAGE 4a - BASELINE: LOGISTIC REGRESSION")
        print("  Purpose: establish the number XGBoost must beat to justify")
        print("  its complexity. A model you can explain to a regulator has")
        print("  real value in a bank.\n")
        n = min(BASELINE_SAMPLE, len(X_train))
        # Take the most RECENT n rows, not a random sample: staying
        # chronological keeps the baseline comparable to the real setup.
        Xb, yb = X_train.tail(n), y_train[-n:]
        base_num = [c for c in spec.numeric if not c.startswith("V")][:60]
        base_cat = [c for c in spec.categorical][:12]
        print(f"  training on {n:,} most recent rows, "
              f"{len(base_num)} numeric + {len(base_cat)} categorical features")

        t0 = time.perf_counter()
        pipe = build_baseline_pipeline(base_num, base_cat)
        pipe.fit(Xb, yb)
        p_val = pipe.predict_proba(X_val)[:, 1]
        print(f"  fitted in {time.perf_counter() - t0:.1f}s")

        base_eval = evaluate_full(y_val, p_val, amt_val, cost_model=cost_model)
        results["baseline_logreg"] = base_eval
        m = base_eval["metrics"]
        print(f"\n  PR-AUC   {m['pr_auc']:.4f}   ({m['pr_auc_lift']:.1f}x random)")
        print(f"  ROC-AUC  {m['roc_auc']:.4f}")
        print(f"  Brier    {m['brier']:.5f}")
        joblib.dump(pipe, models_dir / "baseline_logreg.joblib")
        del pipe
        gc.collect()

    # ---- 6. candidate: XGBoost -------------------------------------------
    section("STAGE 4b - XGBOOST")
    spw = scale_pos_weight(pd.Series(y_train))
    print(f"  scale_pos_weight = {spw:.1f}")
    print("  -> during training, one missed fraud counts as much as")
    print(f"     {spw:.1f} false alarms. This rebalances the loss function")
    print("     WITHOUT inventing synthetic rows the way SMOTE does.\n")

    t0 = time.perf_counter()
    model = build_xgboost(pd.Series(y_train))
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    fit_s = time.perf_counter() - t0
    best_it = getattr(model, "best_iteration", None)
    print(f"\n  fitted in {fit_s:.1f}s")
    print(f"  early stopping chose {best_it} of 600 trees")
    print("  -> validation PR-AUC stopped improving there; more trees would")
    print("     have started memorising the training set.")

    p_val = model.predict_proba(X_val)[:, 1]
    xgb_eval = evaluate_full(y_val, p_val, amt_val, cost_model=cost_model)
    results["xgboost"] = xgb_eval
    results["xgboost"]["best_iteration"] = int(best_it) if best_it is not None else None
    results["xgboost"]["fit_seconds"] = round(fit_s, 1)

    m = xgb_eval["metrics"]
    print(f"\n  PR-AUC   {m['pr_auc']:.4f}   ({m['pr_auc_lift']:.1f}x random)")
    print(f"  ROC-AUC  {m['roc_auc']:.4f}")
    print(f"  Brier    {m['brier']:.5f}")

    joblib.dump(model, models_dir / "xgboost.joblib")
    joblib.dump(fe, models_dir / "frequency_encoder.joblib")
    joblib.dump(features, models_dir / "feature_names.joblib")

    # ---- 7. operating points ---------------------------------------------
    section("STAGE 5 - OPERATING POINTS (the probability -> decision step)")
    print("A model gives a PROBABILITY. A business needs a DECISION.")
    print("Which threshold you pick is a BUSINESS choice, not a statistical one.\n")

    print("--- If the fraud team requires a minimum precision ---")
    print(f"  {'precision':>10} {'recall':>8} {'alerts':>8} {'caught':>8} {'missed':>8}")
    for k, v in xgb_eval["at_precision"].items():
        print(f"  {k:>10} {v['recall']:>8.1%} {v['alert_rate']:>8.2%} "
              f"{v['tp']:>8,} {v['fn']:>8,}")

    print("\n--- If the fraud team has a fixed daily alert budget ---")
    print(f"  {'budget':>10} {'precision':>10} {'recall':>8} {'caught':>8}")
    for k, v in xgb_eval["at_alert_budget"].items():
        print(f"  {k:>10} {v['precision']:>10.1%} {v['recall']:>8.1%} {v['tp']:>8,}")

    # ---- 8. the risk engine ----------------------------------------------
    section("STAGE 5 - RISK ENGINE: choosing the threshold by MONEY")
    co = xgb_eval["cost_optimal"]
    dn = xgb_eval["do_nothing_baseline"]
    va = xgb_eval["value_added"]
    print("Cost model:")
    print(f"  false negative = the transaction amount (we refund the customer)")
    print(f"  false positive = GBP {cost_model.fp_cost:.0f} (review time + friction)")
    print(f"  recovery rate  = {cost_model.tp_recovery_rate:.0%} of caught fraud\n")
    print(f"  do nothing        : GBP {dn['net_cost']:>12,.0f} lost to fraud")
    print(f"  cost-optimal      : GBP {co['net_cost']:>12,.0f} net cost")
    print(f"  NET SAVING        : GBP {va['net_saving']:>12,.0f} "
          f"({va['pct_of_fraud_loss_avoided']:.1f}% of fraud loss avoided)")
    print(f"\n  chosen threshold  : {co['threshold']:.4f}")
    print(f"  precision {co['precision']:.1%} | recall {co['recall']:.1%} "
          f"| alert rate {co['alert_rate']:.2%}")
    print(f"  caught {co['tp']:,} fraud | missed {co['fn']:,} | "
          f"false alarms {co['fp']:,}")

    # ---- 9. persist ------------------------------------------------------
    with open(cfg.reports_dir / "stage04_05_model_results.json", "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    section("DONE")
    print(f"  total runtime  {time.perf_counter() - t_start:.0f}s")
    print(f"  models    -> models/")
    print(f"  results   -> reports/stage04_05_model_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
