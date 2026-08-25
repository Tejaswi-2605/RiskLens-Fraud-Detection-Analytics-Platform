"""Stage 5 (part 2) - probability calibration.

Why this is necessary, and why it is easy to forget
---------------------------------------------------
`scale_pos_weight = 27.5` fixes the model's RANKING under class imbalance, but
it does so by lying to the loss function about how common fraud is. The model
is trained as though roughly half of all transactions were fraud.

The consequence: the model's output is a good *score* but a bad *probability*.
It is systematically OVER-CONFIDENT - our highest alerts come out at 0.999
when their true fraud rate is lower than that.

For ranking, that does not matter at all. For our risk engine it matters
enormously, because the engine computes

    expected loss = probability x amount

If the probability is inflated, every pound figure is wrong even when the
ordering of alerts is perfect.

What calibration is
-------------------
A monotonic function applied AFTER the model that maps raw scores to honest
probabilities. Because it is monotonic it cannot change the ranking - PR-AUC
and ROC-AUC are unchanged by construction. It only fixes the numbers.

    Tiny example.
        Model says 0.90 for 1,000 transactions.
        Only 400 of them are actually fraud.
        A calibrator learns  0.90 -> 0.40.
        Ranking unchanged; the number is now true.

Two methods
-----------
PLATT SCALING (sigmoid)
    Fits a logistic regression on the model's scores: one slope, one
    intercept. Two parameters, so it needs very little data and cannot
    overfit. But it can only apply an S-shaped correction - if the real
    distortion has a different shape, it cannot express it.

ISOTONIC REGRESSION
    Fits any non-decreasing step function. Far more flexible, so it can
    correct distortions Platt cannot. The cost is that it needs more data and
    will happily overfit a small calibration set.

With ~36,000 validation rows and ~1,250 positives we have enough for isotonic,
but we fit BOTH and select on measured Brier score rather than assert which is
better.

The data-splitting trap
-----------------------
The calibrator is a FITTED transformation, so it obeys the same rule as every
other one: it must not be fitted on data used to report performance.

  * Fitting it on TRAIN is wrong - the model already fits train well, so the
    scores there are not representative and the calibrator learns the wrong
    mapping.
  * Fitting it on TEST is leakage.
  * Fitting it on all of VALIDATION and then also choosing the threshold on
    validation double-uses the same rows.

So we split VALIDATION chronologically in two: the earlier half calibrates,
the later half selects the threshold. Test stays sealed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

log = logging.getLogger(__name__)

Method = Literal["isotonic", "sigmoid"]


@dataclass
class CalibrationResult:
    """What calibration achieved, measured rather than assumed."""

    method: str
    brier_before: float
    brier_after: float
    improvement_pct: float
    ece_before: float
    ece_after: float
    max_prob_before: float
    max_prob_after: float
    n_calibration: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "brier_before": round(self.brier_before, 6),
            "brier_after": round(self.brier_after, 6),
            "improvement_pct": round(self.improvement_pct, 2),
            "ece_before": round(self.ece_before, 5),
            "ece_after": round(self.ece_after, 5),
            "max_prob_before": round(self.max_prob_before, 5),
            "max_prob_after": round(self.max_prob_after, 5),
            "n_calibration": self.n_calibration,
        }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """ECE - the headline calibration number, and more readable than Brier.

    Bin predictions by confidence, and in each bin compare the average
    predicted probability against the observed fraud rate. ECE is the
    weighted mean of those gaps.

        Tiny example, 3 bins:
            bin      avg predicted   observed   gap    weight
            0.0-0.1      0.05          0.04     0.01    0.90
            0.1-0.5      0.30          0.22     0.08    0.08
            0.5-1.0      0.85          0.55     0.30    0.02
            ECE = 0.90*0.01 + 0.08*0.08 + 0.02*0.30 = 0.021

    Read it directly: "on average, predictions are off by 2.1 percentage
    points." Brier conflates calibration with discrimination; ECE isolates
    calibration.

    We use QUANTILE bins rather than equal-width bins, because our scores are
    heavily concentrated near zero - equal-width bins would leave the upper
    bins nearly empty and the estimate extremely noisy.
    """
    if len(y_true) == 0:
        return float("nan")

    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float(abs(y_prob.mean() - y_true.mean()))
    edges[0], edges[-1] = -np.inf, np.inf

    idx = np.digitize(y_prob, edges[1:-1], right=True)
    total, n = 0.0, len(y_true)
    for b in np.unique(idx):
        m = idx == b
        if not m.any():
            continue
        total += m.sum() / n * abs(y_prob[m].mean() - y_true[m].mean())
    return float(total)


class ProbabilityCalibrator:
    """Fits a monotonic mapping from model score to honest probability.

    Deliberately NOT sklearn's CalibratedClassifierCV. That wrapper refits the
    base estimator internally via cross-validation, which for us would mean
    retraining XGBoost several times - about 25 minutes each - and would
    cross-validate randomly, breaking the temporal ordering that the whole
    project depends on.

    We already have out-of-sample scores on a held-out chronological slice,
    which is exactly what a calibrator needs. So we fit directly on those.
    """

    def __init__(self, method: Method = "isotonic"):
        self.method = method
        self.model_: Any = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        s = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y).ravel()

        if self.method == "isotonic":
            # out_of_bounds="clip" so a test score outside the calibration
            # range maps to the nearest endpoint instead of raising.
            self.model_ = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True
            ).fit(s, y)
        elif self.method == "sigmoid":
            # Platt scaling: logistic regression on the raw score.
            self.model_ = LogisticRegression(C=1e10, solver="lbfgs").fit(
                s.reshape(-1, 1), y
            )
        else:
            raise ValueError(f"unknown method {self.method!r}")
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float).ravel()
        if self.model_ is None:
            raise RuntimeError("calibrator not fitted")
        if self.method == "isotonic":
            out = self.model_.predict(s)
        else:
            out = self.model_.predict_proba(s.reshape(-1, 1))[:, 1]
        return np.clip(out, 0.0, 1.0)

    __call__ = transform


def fit_and_compare(
    cal_scores: np.ndarray,
    cal_y: np.ndarray,
    eval_scores: np.ndarray,
    eval_y: np.ndarray,
) -> tuple[ProbabilityCalibrator, CalibrationResult, dict[str, float]]:
    """Fit both methods on the calibration slice, pick the better on Brier.

    Selection is made on the EVALUATION slice, not the calibration slice -
    judging a fitted transformation on the data it was fitted to would always
    favour the more flexible method (isotonic), which is precisely the
    overfitting we are trying to detect.
    """
    scores_by_method: dict[str, float] = {}
    fitted: dict[str, ProbabilityCalibrator] = {}

    for method in ("isotonic", "sigmoid"):
        c = ProbabilityCalibrator(method).fit(cal_scores, cal_y)
        p = c.transform(eval_scores)
        scores_by_method[method] = float(brier_score_loss(eval_y, p))
        fitted[method] = c
        log.info("calibration %-9s -> Brier %.6f", method, scores_by_method[method])

    best = min(scores_by_method, key=scores_by_method.get)
    calibrator = fitted[best]
    p_after = calibrator.transform(eval_scores)

    b_before = float(brier_score_loss(eval_y, eval_scores))
    b_after = scores_by_method[best]

    result = CalibrationResult(
        method=best,
        brier_before=b_before,
        brier_after=b_after,
        improvement_pct=100.0 * (b_before - b_after) / b_before if b_before else 0.0,
        ece_before=expected_calibration_error(eval_y, eval_scores),
        ece_after=expected_calibration_error(eval_y, p_after),
        max_prob_before=float(eval_scores.max()),
        max_prob_after=float(p_after.max()),
        n_calibration=int(len(cal_y)),
    )
    log.info("selected %s: Brier %.6f -> %.6f (%.1f%% better)",
             best, b_before, b_after, result.improvement_pct)
    return calibrator, result, scores_by_method


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Bin-by-bin predicted vs observed - the numbers behind a reliability plot.

    A perfectly calibrated model has predicted == observed in every bin.
    """
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return []
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(y_prob, edges[1:-1], right=True)

    rows = []
    for b in np.unique(idx):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": int(b),
            "n": int(m.sum()),
            "mean_predicted": round(float(y_prob[m].mean()), 5),
            "observed_rate": round(float(y_true[m].mean()), 5),
            "gap": round(float(y_prob[m].mean() - y_true[m].mean()), 5),
        })
    return rows
