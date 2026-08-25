"""Stage 5 - evaluation, threshold selection and the risk engine.

The central idea of this file
-----------------------------
A model outputs a PROBABILITY. A business needs a DECISION. Converting one to
the other requires a threshold, and choosing that threshold is a BUSINESS
decision, not a statistical one - it depends on what a mistake costs.

Two mistakes, very different costs:

  FALSE NEGATIVE  - fraud we let through.  Cost = the money lost (the amount).
  FALSE POSITIVE  - a good customer declined. Cost = review effort, plus
                    goodwill, plus the chance they never come back.

Because a false negative costs the *transaction amount* and a false positive
costs a roughly fixed amount, the optimal threshold is not 0.5 and cannot be
found by maximising accuracy or even F1. It must be found by minimising money.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

log = logging.getLogger(__name__)


# =========================================================================
# Core metrics
# =========================================================================
@dataclass
class Metrics:
    """Threshold-free metrics: how good is the RANKING, regardless of cutoff."""

    pr_auc: float
    roc_auc: float
    brier: float
    n: int
    n_positive: int
    base_rate: float
    pr_auc_lift: float

    def as_dict(self) -> dict[str, Any]:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Metrics:
    """Threshold-free scores.

    PR-AUC (average precision) - THE headline metric here
    ----------------------------------------------------
    Area under the precision-recall curve. It only cares about the positive
    class, which is what we want when positives are 3.5% of the data.

    Crucially, a random model scores PR-AUC = the base rate = 0.035. So a
    PR-AUC of 0.70 is 20x better than random. We report that ratio as
    `pr_auc_lift`, because 0.70 means nothing without knowing the baseline.

    ROC-AUC - reported, but NOT the metric we optimise
    --------------------------------------------------
    Probability that a random fraud is ranked above a random legitimate
    transaction. It is *optimistically biased* under heavy imbalance: it
    rewards ranking the 96.5% majority correctly, which is easy. A model can
    look excellent on ROC-AUC and be useless in production.

    Brier score - LOWER is better
    -----------------------------
    Mean squared error of the predicted probabilities. Measures CALIBRATION:
    when the model says 0.3, does fraud actually occur 30% of the time?
    We need this because the risk engine multiplies probability by money.
    """
    base = float(np.mean(y_true))
    pr = float(average_precision_score(y_true, y_prob))
    return Metrics(
        pr_auc=pr,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        n=int(len(y_true)),
        n_positive=int(np.sum(y_true)),
        base_rate=base,
        pr_auc_lift=float(pr / base) if base else float("nan"),
    )


def confusion_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Confusion matrix and the derived rates at one cutoff."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        # What fraction of ALL traffic gets flagged. This is the operational
        # constraint: a fraud team can only review so many alerts a day.
        "alert_rate": round(float((tp + fp) / len(y_true)), 4),
    }


# =========================================================================
# Threshold selection
# =========================================================================
def threshold_for_precision(
    y_true: np.ndarray, y_prob: np.ndarray, target_precision: float
) -> float:
    """Lowest threshold that still achieves the required precision.

    Why a fraud team asks for this
    ------------------------------
    Analysts have finite time. If precision is 20%, four of every five alerts
    they investigate are innocent customers, and they stop trusting the
    system. So the business says "I need at least 50% precision" and we find
    the threshold that delivers it while catching as much fraud as possible.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall have one more element than thresholds - drop the last.
    ok = np.where(precision[:-1] >= target_precision)[0]
    if len(ok) == 0:
        log.warning("precision %.2f unreachable; returning max threshold", target_precision)
        return float(thresholds[-1])
    return float(thresholds[ok[0]])


def threshold_for_alert_budget(y_prob: np.ndarray, budget_rate: float) -> float:
    """Threshold that flags exactly `budget_rate` of all traffic.

    The most operationally realistic constraint of all: "my team can review
    500 alerts a day out of 50,000 transactions" -> budget_rate = 1%.
    We simply take the (1 - budget) quantile of the scores.
    """
    return float(np.quantile(y_prob, 1.0 - budget_rate))


# =========================================================================
# The risk engine - money, not metrics
# =========================================================================
@dataclass
class CostModel:
    """What each type of mistake costs, in currency.

    Defaults are deliberately explicit and arguable - in a real engagement
    these numbers come from the business, not from the data scientist.

    fn_cost_is_amount
        A missed fraud costs the transaction value: the bank refunds the
        customer and eats the loss. So the cost VARIES per transaction, which
        is exactly why a single global threshold is suboptimal.

    fp_cost
        A false positive costs analyst review time plus customer friction.
        Fixed per alert. GBP 15 is a common order of magnitude for a manual
        review; the goodwill cost of a wrongly declined payment is larger but
        harder to quantify.

    tp_recovery_rate
        Catching fraud does not always recover 100% of the value - some is
        already gone. 0.9 is a reasonable assumption.
    """

    fp_cost: float = 15.0
    tp_recovery_rate: float = 0.9
    fn_cost_is_amount: bool = True

    def expected_cost(
        self, y_true: np.ndarray, y_prob: np.ndarray, amounts: np.ndarray, threshold: float
    ) -> dict[str, float]:
        """Total money LOST at this threshold. Lower is better. Never negative.

        The formulation matters, and getting it wrong is a real trap
        -----------------------------------------------------------
        Catching fraud AVOIDS a loss; it does not EARN revenue. An earlier
        version of this function computed

            cost = fn_loss + fp_cost - recovered          # WRONG

        which treats every caught fraud as income. Under that formula the
        optimiser is rewarded for flagging more, so it drives the threshold
        down until it flags everything, and total "cost" goes NEGATIVE - the
        model appears to print money. It also produced the impossible headline
        "116.8% of fraud loss avoided".

        The correct accounting asks: of the money at risk, how much do we
        still lose after acting?

            cost = fraud we MISSED (full value)
                 + fraud we CAUGHT but could not recover
                 + review cost of the good customers we flagged

        Sanity checks that this formulation satisfies and the old one did not:

          * threshold = 1.0 (flag nothing)  -> cost == total fraud value,
            which is exactly the do-nothing baseline.
          * cost >= 0 always.
          * saving <= 100% of the fraud loss, always.
        """
        flagged = y_prob >= threshold
        is_fraud = y_true.astype(bool)

        # Fraud we caught. We recover most of the value, but not all - some has
        # already moved. The UNRECOVERED remainder is a real loss.
        tp_amount = float(amounts[flagged & is_fraud].sum())
        recovered = tp_amount * self.tp_recovery_rate
        unrecovered = tp_amount - recovered

        # Fraud we missed: full loss.
        fn_loss = float(amounts[~flagged & is_fraud].sum())

        # Good customers we flagged: fixed friction cost each.
        fp_count = int((flagged & ~is_fraud).sum())
        fp_loss = fp_count * self.fp_cost

        total = fn_loss + unrecovered + fp_loss
        return {
            "threshold": round(float(threshold), 6),
            "fraud_caught_value": round(tp_amount, 2),
            "recovered": round(recovered, 2),
            "unrecovered_on_caught": round(unrecovered, 2),
            "fraud_missed_value": round(fn_loss, 2),
            "false_positive_count": fp_count,
            "false_positive_cost": round(fp_loss, 2),
            "net_cost": round(total, 2),
        }

    def optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        amounts: np.ndarray,
        n_grid: int = 200,
    ) -> tuple[float, pd.DataFrame]:
        """Sweep thresholds and pick the one that minimises net cost.

        This is the step that makes the project a *risk* system rather than a
        *classification* exercise. We are not choosing a threshold to maximise
        F1; we are choosing it to lose the least money.

        The grid comes from score quantiles rather than a uniform 0..1 range,
        because predicted probabilities are extremely concentrated near zero -
        a uniform grid would waste most of its points in empty space.
        """
        grid = np.unique(np.quantile(y_prob, np.linspace(0.50, 0.9995, n_grid)))
        rows = [self.expected_cost(y_true, y_prob, amounts, t) for t in grid]
        table = pd.DataFrame(rows)
        best = float(table.loc[table["net_cost"].idxmin(), "threshold"])
        return best, table


def evaluate_full(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    *,
    cost_model: CostModel | None = None,
    precision_targets: tuple[float, ...] = (0.5, 0.7, 0.9),
    alert_budgets: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
) -> dict[str, Any]:
    """The complete evaluation: ranking quality, operating points, and money."""
    cm = cost_model or CostModel()
    out: dict[str, Any] = {"metrics": compute_metrics(y_true, y_prob).as_dict()}

    out["at_precision"] = {
        f"{p:.0%}": confusion_at(y_true, y_prob, threshold_for_precision(y_true, y_prob, p))
        for p in precision_targets
    }
    out["at_alert_budget"] = {
        f"{b:.1%}": confusion_at(y_true, y_prob, threshold_for_alert_budget(y_prob, b))
        for b in alert_budgets
    }

    best_t, sweep = cm.optimal_threshold(y_true, y_prob, amounts)
    out["cost_optimal"] = {
        **cm.expected_cost(y_true, y_prob, amounts, best_t),
        **confusion_at(y_true, y_prob, best_t),
    }
    # Baseline for comparison: what if we blocked nothing at all?
    out["do_nothing_baseline"] = {
        "net_cost": round(float(amounts[y_true.astype(bool)].sum()), 2),
        "note": "all fraud goes through; no false positives",
    }
    saved = out["do_nothing_baseline"]["net_cost"] - out["cost_optimal"]["net_cost"]
    out["value_added"] = {
        "net_saving": round(saved, 2),
        "pct_of_fraud_loss_avoided": round(
            100.0 * saved / out["do_nothing_baseline"]["net_cost"], 2
        ) if out["do_nothing_baseline"]["net_cost"] else 0.0,
    }
    return out
