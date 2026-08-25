"""Stage 5 (part 2) - calibrate the model, then re-price the risk engine.

The problem
-----------
`scale_pos_weight = 27.5` fixes the model's RANKING under 3.5% imbalance, but
it does so by training as though roughly half of all transactions were fraud.
The model therefore produces good SCORES and bad PROBABILITIES - it is
systematically over-confident. Our top alert scored 0.9999.

For ranking that is harmless. For the risk engine it is not, because the
engine computes

    expected loss = probability x amount

An inflated probability makes every pound figure wrong even when the ordering
of alerts is perfect.

The data-splitting discipline
-----------------------------
A calibrator is a FITTED transformation, so it obeys the same rule as every
other one in this project: it must not be fitted on data used to report
performance.

  * Fitting on TRAIN is wrong - the model already fits train well, so scores
    there are unrepresentative and the calibrator learns the wrong mapping.
  * Fitting on TEST is leakage.
  * Fitting on ALL of validation and then also selecting the threshold there
    double-uses the same rows.

So VALIDATION is split chronologically in two:

    val_early  ->  fit the calibrator
    val_late   ->  choose between isotonic and Platt, and pick the threshold
    test       ->  report once, untouched

Chronologically rather than randomly, for the same reason the main split is
temporal: a random half would mix future and past.

Usage
-----
    python scripts/run_calibrate.py
"""

from __future__ import annotations

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
from risklens.features.build import add_deterministic_features  # noqa: E402
from risklens.features.entity import build_entity_features  # noqa: E402
from risklens.models.calibrate import (  # noqa: E402
    count_inversions, fit_and_compare, reliability_table,
)
from risklens.models.evaluate import (  # noqa: E402
    CostModel, compute_metrics, confusion_at, evaluate_full,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("stage05b")


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n  {t}\n{'=' * 74}", flush=True)


def main() -> int:
    cfg = load_data_config()
    md = cfg.root / "models"
    t_start = time.perf_counter()

    for f in ("xgboost.joblib", "frequency_encoder.joblib", "feature_names.joblib"):
        if not (md / f).is_file():
            log.error("missing models/%s", f)
            return 1

    model = joblib.load(md / "xgboost.joblib")
    encoder = joblib.load(md / "frequency_encoder.joblib")
    features = joblib.load(md / "feature_names.joblib")

    log.info("rebuilding features ...")
    df = load_joined(cfg)
    df = add_deterministic_features(df)
    df = build_entity_features(df, time_col=cfg.time_column,
                               amount_col=cfg.amount_column)
    df = encoder.transform(df)
    masks, _, _ = temporal_split(df, time_col=cfg.time_column,
                                 target_col=cfg.target, split_cfg=cfg.split)

    val = df.loc[masks["val"]].sort_values(cfg.time_column)
    test = df.loc[masks["test"]].sort_values(cfg.time_column)
    del df

    # ---- chronological halving of validation ----------------------------
    cut = len(val) // 2
    val_early, val_late = val.iloc[:cut], val.iloc[cut:]

    section("SPLIT FOR CALIBRATION")
    print("Validation is halved CHRONOLOGICALLY, not randomly - a random half")
    print("would mix future and past, which is what the whole project avoids.\n")
    for name, part in [("val_early (fit calibrator)", val_early),
                       ("val_late  (select + threshold)", val_late),
                       ("test      (report once)", test)]:
        y = part[cfg.target]
        print(f"  {name:<32} {len(part):>7,} rows   fraud {int(y.sum()):>5,} "
              f"({y.mean():.3%})")

    p_early = model.predict_proba(val_early[features])[:, 1]
    p_late = model.predict_proba(val_late[features])[:, 1]
    p_test = model.predict_proba(test[features])[:, 1]
    y_early = val_early[cfg.target].to_numpy()
    y_late = val_late[cfg.target].to_numpy()
    y_test = test[cfg.target].to_numpy()
    amt_late = val_late[cfg.amount_column].to_numpy()
    amt_test = test[cfg.amount_column].to_numpy()

    # ---- how bad is it before calibration? ------------------------------
    section("BEFORE CALIBRATION - the over-confidence, quantified")
    print("A calibrated model's predicted probability should match the")
    print("observed fraud rate in each bin. Ours does not.\n")
    before = reliability_table(y_late, p_late, n_bins=10)
    print(f"  {'bin':>4} {'n':>8} {'predicted':>11} {'observed':>10} {'gap':>9}")
    for r in before:
        print(f"  {r['bin']:>4} {r['n']:>8,} {r['mean_predicted']:>11.4f} "
              f"{r['observed_rate']:>10.4f} {r['gap']:>+9.4f}")
    print(f"\n  max predicted probability: {p_late.max():.5f}")
    print("  A positive gap in the top bins means over-confidence: the model")
    print("  claims more certainty than the outcomes support.")

    # ---- fit and choose --------------------------------------------------
    section("FITTING CALIBRATORS (isotonic vs Platt)")
    print("ISOTONIC   any non-decreasing step function. Flexible, needs data.")
    print("PLATT      a logistic curve - two parameters. Cannot overfit, but")
    print("           can only apply an S-shaped correction.\n")
    print("Selection happens on val_late, NOT on the data they were fitted to -")
    print("judging a fitted transform on its own training data would always")
    print("favour the more flexible one, which is the overfitting we want to")
    print("detect.\n")

    calibrator, result, per_method = fit_and_compare(
        p_early, y_early, p_late, y_late
    )
    print()
    print(f"  {'method':<10} {'Brier':>10} {'ECE':>10} {'distinct':>10} "
          f"{'inversions':>11}")
    for m in ("isotonic", "sigmoid"):
        st = per_method[m]
        mark = "  <- selected" if m == result.method else ""
        print(f"  {m:<10} {st['brier']:>10.6f} {st['ece']:>10.5f} "
              f"{st['distinct_values']:>10,} {st['inversions']:>11}{mark}")
    print(f"\n  {per_method['_selection']['reason']}")
    print()
    print("  NOTE the 'distinct' column. Isotonic is a STEP function, so it")
    print("  collapses many raw scores into one value. That costs real")
    print("  granularity: with few distinct scores you cannot place a")
    print("  threshold at an arbitrary alert budget, because whole blocks of")
    print("  transactions share a single score.")

    p_late_cal = calibrator.transform(p_late)
    p_test_cal = calibrator.transform(p_test)

    section("AFTER CALIBRATION")
    r = result.as_dict()
    print(f"  method                {r['method']}")
    print(f"  Brier      {r['brier_before']:.6f}  ->  {r['brier_after']:.6f}   "
          f"({r['improvement_pct']:+.1f}%)")
    print(f"  ECE        {r['ece_before']:.5f}  ->  {r['ece_after']:.5f}")
    print(f"  max prob   {r['max_prob_before']:.5f}  ->  {r['max_prob_after']:.5f}")
    print()
    print("  ECE is 'on average, predictions are off by this many percentage")
    print("  points'. Brier conflates calibration with discrimination; ECE")
    print("  isolates calibration, which is what we are fixing.")

    after = reliability_table(y_late, p_late_cal, n_bins=10)
    print(f"\n  {'bin':>4} {'n':>8} {'predicted':>11} {'observed':>10} {'gap':>9}")
    for row in after:
        print(f"  {row['bin']:>4} {row['n']:>8,} {row['mean_predicted']:>11.4f} "
              f"{row['observed_rate']:>10.4f} {row['gap']:>+9.4f}")

    # ---- ranking must be unchanged --------------------------------------
    section("SANITY CHECK - did the calibrator reorder anything?")
    m_raw = compute_metrics(y_late, p_late)
    m_cal = compute_metrics(y_late, p_late_cal)
    inv = count_inversions(p_late, p_late_cal)

    print("The right test is INVERSIONS, not 'PR-AUC must be identical'.")
    print()
    print("An earlier version of this script checked PR-AUC equality and")
    print("reported a spurious FAIL on a perfectly correct isotonic")
    print("calibrator. Isotonic is a step function: it never reorders, but it")
    print("creates enormous numbers of TIES, and average_precision treats tied")
    print("scores differently from distinct ones. ROC-AUC barely moves because")
    print("it averages over ties.")
    print()
    print("So the honest check is: sort by raw score, and confirm the")
    print("calibrated score never DECREASES. Equal is fine; lower is a real")
    print("inversion.")
    print()
    print(f"  strict inversions (real reordering)   {inv}")
    print(f"  PR-AUC    {m_raw.pr_auc:.5f}  ->  {m_cal.pr_auc:.5f}")
    print(f"  ROC-AUC   {m_raw.roc_auc:.5f}  ->  {m_cal.roc_auc:.5f}")
    ok = inv == 0
    print(f"\n  {'PASS - zero inversions, ranking preserved' if ok else 'FAIL - genuine reordering!'}")

    # ---- re-price the risk engine ---------------------------------------
    section("RISK ENGINE, RE-PRICED ON HONEST PROBABILITIES")
    cm = CostModel()
    raw_eval = evaluate_full(y_late, p_late, amt_late, cost_model=cm)
    cal_eval = evaluate_full(y_late, p_late_cal, amt_late, cost_model=cm)

    print(f"  {'':<22} {'raw':>14} {'calibrated':>14}")
    for label, key in [("threshold", "threshold"), ("precision", "precision"),
                       ("recall", "recall"), ("alert rate", "alert_rate")]:
        a, b = raw_eval["cost_optimal"][key], cal_eval["cost_optimal"][key]
        fmt = ".4f" if label == "threshold" else ".2%"
        print(f"  {label:<22} {a:>14{fmt}} {b:>14{fmt}}")
    for label, key in [("net cost", "net_cost")]:
        a, b = raw_eval["cost_optimal"][key], cal_eval["cost_optimal"][key]
        print(f"  {label:<22} {a:>14,.0f} {b:>14,.0f}")
    a = raw_eval["value_added"]["pct_of_fraud_loss_avoided"]
    b = cal_eval["value_added"]["pct_of_fraud_loss_avoided"]
    print(f"  {'% loss avoided':<22} {a:>13.1f}% {b:>13.1f}%")
    print()
    print("  The chosen threshold moves because the probability SCALE changed.")
    print("  The underlying decisions are similar - the ranking is identical -")
    print("  but the expected-loss arithmetic is now built on probabilities")
    print("  that mean what they say.")

    # ---- final test report ----------------------------------------------
    section("TEST - final, with calibrated probabilities")
    chosen = float(cal_eval["cost_optimal"]["threshold"])
    tm = compute_metrics(y_test, p_test_cal)
    conf = confusion_at(y_test, p_test_cal, chosen)
    cost = cm.expected_cost(y_test, p_test_cal, amt_test, chosen)
    dn = float(amt_test[y_test.astype(bool)].sum())
    saving = dn - cost["net_cost"]

    print(f"  PR-AUC   {tm.pr_auc:.4f}   ({tm.pr_auc_lift:.1f}x random)")
    print(f"  ROC-AUC  {tm.roc_auc:.4f}")
    print(f"  Brier    {tm.brier:.5f}")
    print(f"\n  at the val_late-chosen threshold {chosen:.4f}:")
    print(f"    precision {conf['precision']:.1%} | recall {conf['recall']:.1%} "
          f"| alert rate {conf['alert_rate']:.2%}")
    print(f"    caught {conf['tp']:,} | missed {conf['fn']:,} "
          f"| false alarms {conf['fp']:,}")
    print(f"\n  do nothing     : GBP {dn:>12,.0f}")
    print(f"  with RiskLens  : GBP {cost['net_cost']:>12,.0f}")
    print(f"  NET SAVING     : GBP {saving:>12,.0f}   "
          f"({100 * saving / dn:.1f}% of fraud loss avoided)")

    # ---- persist ---------------------------------------------------------
    joblib.dump(calibrator, md / "calibrator.joblib")
    payload = {
        "calibration": result.as_dict(),
        "per_method": per_method,
        "reliability_before": before,
        "reliability_after": after,
        "ranking_preserved": {
            "strict_inversions": inv,
            "pr_auc_raw": round(m_raw.pr_auc, 5),
            "pr_auc_calibrated": round(m_cal.pr_auc, 5),
            "note": "PR-AUC can move under isotonic because of TIES, not "
                    "reordering. Inversions is the correct test.",
            "passed": bool(ok),
        },
        "risk_engine_raw": raw_eval["cost_optimal"],
        "risk_engine_calibrated": cal_eval["cost_optimal"],
        "test": {
            "metrics": tm.as_dict(),
            "threshold": chosen,
            "confusion": conf,
            "cost": cost,
            "do_nothing_cost": round(dn, 2),
            "net_saving": round(saving, 2),
            "pct_of_fraud_loss_avoided": round(100 * saving / dn, 2),
        },
    }
    with open(cfg.reports_dir / "stage05b_calibration.json", "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    section("DONE")
    print(f"  runtime {time.perf_counter() - t_start:.0f}s")
    print("  models/calibrator.joblib")
    print("  reports/stage05b_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
