"""Stage 5 - evaluation and the final, honest test-set report.

Why this is a separate script from training
-------------------------------------------
Training is expensive (~25 minutes). Evaluation is seconds. Separating them
means a change to the cost model or the threshold policy can be re-run without
retraining - which is exactly what happened: the first cost function treated
caught fraud as revenue rather than as avoided loss, and re-running took
seconds rather than half an hour.

The methodology this script enforces
------------------------------------
    VALIDATION  ->  choose the threshold          (used many times)
    TEST        ->  report the final number       (used ONCE)

That order is the whole point of a three-way split. If you tune a threshold on
test and then report test performance, the number is optimistically biased -
you have fitted a parameter to the data you are reporting on, and no code can
detect it afterwards.

So the threshold is selected on validation and then APPLIED, unchanged, to
test. Whatever test says is what we report.

Usage
-----
    python scripts/run_eval.py
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

from risklens.config import load_data_config  # noqa: E402
from risklens.data.ingest import load_joined  # noqa: E402
from risklens.data.split import temporal_split  # noqa: E402
from risklens.features.build import add_deterministic_features  # noqa: E402
from risklens.features.entity import build_entity_features  # noqa: E402
from risklens.models.evaluate import (  # noqa: E402
    CostModel, confusion_at, compute_metrics, evaluate_full,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("stage05")


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n  {t}\n{'=' * 74}", flush=True)


def main() -> int:
    cfg = load_data_config()
    md = cfg.root / "models"
    t_start = time.perf_counter()

    for f in ("xgboost.joblib", "frequency_encoder.joblib", "feature_names.joblib"):
        if not (md / f).is_file():
            log.error("missing %s - run scripts/run_train.py first", f)
            return 1

    model = joblib.load(md / "xgboost.joblib")
    encoder = joblib.load(md / "frequency_encoder.joblib")
    features = joblib.load(md / "feature_names.joblib")

    # Rebuild features using the SAME code path and the SAME fitted encoder
    # the training run used. This is also a live check against
    # training/serving skew.
    log.info("rebuilding features ...")
    df = load_joined(cfg)
    df = add_deterministic_features(df)
    df = build_entity_features(df, time_col=cfg.time_column,
                               amount_col=cfg.amount_column)
    df = encoder.transform(df)
    masks, _, split_summary = temporal_split(
        df, time_col=cfg.time_column, target_col=cfg.target, split_cfg=cfg.split
    )

    parts = {}
    for name in ("val", "test"):
        sub = df[masks[name]]
        parts[name] = {
            "X": sub[features],
            "y": sub[cfg.target].to_numpy(),
            "amt": sub[cfg.amount_column].to_numpy(),
        }
    del df

    cost_model = CostModel()
    results: dict = {"split": split_summary, "cost_model": {
        "fp_cost": cost_model.fp_cost,
        "tp_recovery_rate": cost_model.tp_recovery_rate,
        "fn_cost": "the transaction amount",
    }}

    # =====================================================================
    # VALIDATION - choose the threshold
    # =====================================================================
    section("VALIDATION - ranking quality and threshold selection")
    p_val = model.predict_proba(parts["val"]["X"])[:, 1]
    val_eval = evaluate_full(
        parts["val"]["y"], p_val, parts["val"]["amt"], cost_model=cost_model
    )
    results["validation"] = val_eval

    m = val_eval["metrics"]
    print(f"  PR-AUC   {m['pr_auc']:.4f}   ({m['pr_auc_lift']:.1f}x random)")
    print(f"  ROC-AUC  {m['roc_auc']:.4f}")
    print(f"  Brier    {m['brier']:.5f}   (lower = better calibrated)")
    print(f"  base rate {m['base_rate']:.4f}  <- what a random model scores on PR-AUC")

    print("\n--- operating points: minimum precision required ---")
    print(f"  {'precision':>10} {'recall':>8} {'alerts':>8} {'caught':>8} {'missed':>8}")
    for k, v in val_eval["at_precision"].items():
        print(f"  {k:>10} {v['recall']:>8.1%} {v['alert_rate']:>8.2%} "
              f"{v['tp']:>8,} {v['fn']:>8,}")

    print("\n--- operating points: fixed alert budget ---")
    print(f"  {'budget':>10} {'precision':>10} {'recall':>8} {'caught':>8}")
    for k, v in val_eval["at_alert_budget"].items():
        print(f"  {k:>10} {v['precision']:>10.1%} {v['recall']:>8.1%} {v['tp']:>8,}")

    section("RISK ENGINE - threshold chosen by MONEY (on validation)")
    print("Cost model:")
    print("  false negative = the transaction amount (we refund the customer)")
    print(f"  false positive = GBP {cost_model.fp_cost:.0f} (review time + friction)")
    print(f"  true positive  = we recover {cost_model.tp_recovery_rate:.0%}; "
          "the remainder is still a loss\n")

    co = val_eval["cost_optimal"]
    dn = val_eval["do_nothing_baseline"]
    va = val_eval["value_added"]
    print(f"  do nothing        : GBP {dn['net_cost']:>12,.0f}   all fraud gets through")
    print(f"  cost-optimal      : GBP {co['net_cost']:>12,.0f}   net loss after acting")
    print(f"  NET SAVING        : GBP {va['net_saving']:>12,.0f}   "
          f"({va['pct_of_fraud_loss_avoided']:.1f}% of fraud loss avoided)")
    print(f"\n  chosen threshold  : {co['threshold']:.4f}")
    print(f"  precision {co['precision']:.1%} | recall {co['recall']:.1%} "
          f"| alert rate {co['alert_rate']:.2%}")
    print(f"  caught {co['tp']:,} | missed {co['fn']:,} | false alarms {co['fp']:,}")

    chosen = float(co["threshold"])

    # =====================================================================
    # TEST - report once, at the threshold chosen on validation
    # =====================================================================
    section("TEST - the final honest number (threshold NOT retuned here)")
    print("The threshold above was selected on VALIDATION. It is applied")
    print("unchanged to TEST. Retuning it here would make the number")
    print("optimistically biased, and nothing could detect that afterwards.\n")

    p_test = model.predict_proba(parts["test"]["X"])[:, 1]
    y_test, amt_test = parts["test"]["y"], parts["test"]["amt"]

    tm = compute_metrics(y_test, p_test)
    conf = confusion_at(y_test, p_test, chosen)
    cost = cost_model.expected_cost(y_test, p_test, amt_test, chosen)
    dn_test = float(amt_test[y_test.astype(bool)].sum())
    saving = dn_test - cost["net_cost"]

    print(f"  PR-AUC   {tm.pr_auc:.4f}   ({tm.pr_auc_lift:.1f}x random)")
    print(f"  ROC-AUC  {tm.roc_auc:.4f}")
    print(f"  Brier    {tm.brier:.5f}")
    print(f"\n  at the validation-chosen threshold {chosen:.4f}:")
    print(f"    precision {conf['precision']:.1%} | recall {conf['recall']:.1%} "
          f"| alert rate {conf['alert_rate']:.2%}")
    print(f"    caught {conf['tp']:,} | missed {conf['fn']:,} "
          f"| false alarms {conf['fp']:,}")
    print(f"\n  do nothing        : GBP {dn_test:>12,.0f}")
    print(f"  with RiskLens     : GBP {cost['net_cost']:>12,.0f}")
    print(f"  NET SAVING        : GBP {saving:>12,.0f}   "
          f"({100 * saving / dn_test:.1f}% of fraud loss avoided)")

    results["test"] = {
        "metrics": tm.as_dict(),
        "threshold_from_validation": chosen,
        "confusion": conf,
        "cost": cost,
        "do_nothing_cost": round(dn_test, 2),
        "net_saving": round(saving, 2),
        "pct_of_fraud_loss_avoided": round(100 * saving / dn_test, 2),
    }

    # Generalisation gap: how much worse is test than validation?
    gap = tm.pr_auc - val_eval["metrics"]["pr_auc"]
    print(f"\n  generalisation gap (test PR-AUC - val PR-AUC): {gap:+.4f}")
    print("  A small gap means the temporal split held and the model is not")
    print("  overfitted to the validation period.")
    results["generalisation_gap_pr_auc"] = round(float(gap), 5)

    with open(cfg.reports_dir / "stage05_evaluation.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    # keep the combined artefact the dev log and API read from in sync
    combined_path = cfg.reports_dir / "stage04_05_model_results.json"
    if combined_path.is_file():
        combined = json.loads(combined_path.read_text(encoding="utf-8"))
        combined["xgboost"].update({
            k: val_eval[k] for k in
            ("cost_optimal", "do_nothing_baseline", "value_added")
        })
        combined["test"] = results["test"]
        combined["generalisation_gap_pr_auc"] = results["generalisation_gap_pr_auc"]
        combined_path.write_text(json.dumps(combined, indent=2, default=str),
                                 encoding="utf-8")

    section("DONE")
    print(f"  runtime {time.perf_counter() - t_start:.0f}s")
    print("  results -> reports/stage05_evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
