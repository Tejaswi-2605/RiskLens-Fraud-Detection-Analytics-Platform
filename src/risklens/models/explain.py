"""Stage 7 - explainability with SHAP.

Why explainability is not optional here
---------------------------------------
Three separate reasons, and only one of them is technical:

  REGULATORY   Under GDPR and equivalent financial regulation, a customer
               declined by an automated system can demand to know why.
               "The gradient boosting model said so" is not an answer.

  OPERATIONAL  A fraud analyst receiving an alert needs to know WHICH
               features triggered it, or they cannot investigate efficiently.

  DEBUGGING    The fastest way to find leakage is to look at what the model
               considers important. A feature with implausibly dominant
               importance is a leak until proven otherwise.

What SHAP is
------------
SHAP = SHapley Additive exPlanations. It borrows the Shapley value from
cooperative game theory: if several players cooperate to earn a payout, how
much did each player contribute?

Here the "players" are features and the "payout" is the prediction. SHAP
answers: how much did each feature push this prediction away from the average?

The property that makes it trustworthy is ADDITIVITY:

    base_value + sum(shap_values) = the actual prediction

So the explanation always sums exactly to the model's output. It cannot be
hand-wavy, and it cannot omit a contributing factor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ReasonCode:
    """One human-readable driver of a single prediction.

    This is the unit an analyst actually consumes, and in Stage 9 it is what
    the LLM turns into a narrative.
    """

    feature: str
    value: Any
    shap_value: float
    direction: str  # "increases risk" | "decreases risk"

    def as_text(self) -> str:
        arrow = "^" if self.shap_value > 0 else "v"
        return (
            f"{arrow} {self.feature} = {self.value} "
            f"({self.direction}, impact {self.shap_value:+.3f})"
        )


def compute_shap_values(model, X: pd.DataFrame, max_rows: int = 5_000):
    """SHAP values for a tree model.

    Why TreeExplainer rather than the generic KernelExplainer
    ---------------------------------------------------------
    KernelExplainer is model-agnostic but approximates by sampling, and is
    orders of magnitude slower - it would need thousands of model evaluations
    per row.

    TreeExplainer exploits the structure of a decision tree to compute EXACT
    Shapley values in polynomial time. For a tree ensemble it is both faster
    and more accurate. Always use it when the model is a tree.

    We subsample rows because we only need a representative picture for the
    GLOBAL view; per-alert explanations are computed on demand for one row.
    """
    import shap

    Xs = X.sample(min(max_rows, len(X)), random_state=42) if len(X) > max_rows else X
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xs)
    log.info("computed SHAP for %s rows x %s features", f"{len(Xs):,}", Xs.shape[1])
    return explainer, values, Xs


def global_importance(shap_values: np.ndarray, X: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    """Global feature importance = MEAN ABSOLUTE SHAP value.

    Why mean ABSOLUTE
    -----------------
    A feature that pushes risk strongly up for some transactions and strongly
    down for others is very important. Averaging the signed values would
    cancel those out to nearly zero and hide it. We want magnitude of
    influence, not net direction.

    Why this beats XGBoost's built-in `feature_importances_`
    --------------------------------------------------------
    The default "gain" importance is computed on the TRAINING data and is
    biased toward high-cardinality features, which get more chances to split.
    SHAP is computed on whatever data you hand it (we use validation) and is
    consistent - if a model relies on a feature more, its SHAP importance
    cannot go down. Gain importance has no such guarantee.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def explain_one(
    explainer,
    model,
    row: pd.DataFrame,
    *,
    top_k: int = 6,
) -> tuple[float, float, list[ReasonCode]]:
    """Explain ONE transaction - the per-alert view a fraud analyst sees.

    Returns (probability, base_value, reason_codes).

    The base_value is the model's average output before any feature is
    considered. Each reason code then says how far one feature moved this
    particular prediction from that average, and in which direction.
    """
    shap_row = explainer.shap_values(row)
    if shap_row.ndim > 1:
        shap_row = shap_row[0]

    base = float(np.ravel(explainer.expected_value)[0])
    prob = float(model.predict_proba(row)[:, 1][0])

    order = np.argsort(np.abs(shap_row))[::-1][:top_k]
    reasons = [
        ReasonCode(
            feature=str(row.columns[i]),
            value=row.iloc[0, i],
            shap_value=float(shap_row[i]),
            direction="increases risk" if shap_row[i] > 0 else "decreases risk",
        )
        for i in order
    ]
    return prob, base, reasons


def leakage_audit(importance: pd.DataFrame, threshold: float = 0.35) -> dict[str, Any]:
    """Flag suspiciously dominant features.

    The heuristic: if ONE feature accounts for more than `threshold` of all
    SHAP magnitude, treat it as a leakage suspect until you can explain it.

    Real fraud signal is diffuse - Stage 2 found nothing with an effect size
    above about 0.24. A single overwhelming feature usually means it encodes
    the answer, the split, or the time period.
    """
    total = float(importance["mean_abs_shap"].sum())
    if total <= 0:
        return {"suspects": [], "note": "no signal"}
    share = importance["mean_abs_shap"] / total
    suspects = importance.loc[share > threshold, "feature"].tolist()
    return {
        "top_feature": importance.iloc[0]["feature"],
        "top_feature_share": round(float(share.iloc[0]), 4),
        "suspects": suspects,
        "verdict": (
            "LEAKAGE SUSPECT - one feature dominates; verify it is knowable "
            "at prediction time" if suspects else
            "healthy - importance is spread across many features, which is "
            "what genuine fraud signal looks like"
        ),
    }
