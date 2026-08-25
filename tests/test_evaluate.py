"""Tests for the risk engine.

Why these exist
---------------
The first version of `CostModel.expected_cost` subtracted recovered fraud as
if catching it EARNED money rather than AVOIDED a loss. The result was a
negative net cost and the impossible headline "116.8% of fraud loss avoided".

Nothing crashed. The metrics looked spectacular. That is exactly the class of
bug that reaches production, so every invariant it violated is now a test.
"""

from __future__ import annotations

import numpy as np
import pytest

from risklens.models.evaluate import (
    CostModel,
    compute_metrics,
    confusion_at,
    evaluate_full,
    threshold_for_alert_budget,
    threshold_for_precision,
)

RNG = np.random.default_rng(0)


def make_case(n: int = 4_000, rate: float = 0.035):
    """Scores correlated with truth, so the model is better than random."""
    y = (RNG.random(n) < rate).astype(int)
    # fraud scores skew high, legit scores skew low, with real overlap
    p = np.clip(RNG.beta(2, 6, n) + y * RNG.uniform(0.15, 0.5, n), 0, 1)
    amounts = np.round(RNG.lognormal(4.2, 1.1, n), 2)
    return y, p, amounts


# ------------------------------------------------------- cost invariants ---
def test_cost_is_never_negative():
    """You cannot make money by declining transactions."""
    y, p, amt = make_case()
    cm = CostModel()
    for t in np.linspace(0.01, 0.99, 40):
        assert cm.expected_cost(y, p, amt, t)["net_cost"] >= 0


def test_flagging_nothing_equals_the_do_nothing_baseline():
    """At a threshold above every score, cost == total fraud value.

    This is the anchor the broken version failed: it returned a cost below
    the baseline even when nothing was flagged.
    """
    y, p, amt = make_case()
    cm = CostModel()
    cost = cm.expected_cost(y, p, amt, threshold=1.01)
    total_fraud = float(amt[y.astype(bool)].sum())
    assert cost["net_cost"] == pytest.approx(total_fraud, rel=1e-9)
    assert cost["false_positive_count"] == 0


def test_flagging_everything_costs_unrecovered_fraud_plus_all_reviews():
    """At threshold 0 we catch all fraud but pay to review every good customer."""
    y, p, amt = make_case()
    cm = CostModel()
    cost = cm.expected_cost(y, p, amt, threshold=0.0)
    total_fraud = float(amt[y.astype(bool)].sum())
    n_legit = int((y == 0).sum())
    expected = total_fraud * (1 - cm.tp_recovery_rate) + n_legit * cm.fp_cost
    assert cost["net_cost"] == pytest.approx(expected, rel=1e-9)
    assert cost["fraud_missed_value"] == pytest.approx(0.0)


def test_saving_cannot_exceed_the_fraud_loss():
    """The impossible '116.8% of fraud loss avoided' must be unreachable."""
    y, p, amt = make_case()
    out = evaluate_full(y, p, amt)
    pct = out["value_added"]["pct_of_fraud_loss_avoided"]
    assert pct <= 100.0
    assert out["cost_optimal"]["net_cost"] >= 0


def test_optimal_threshold_beats_both_extremes():
    """A sensible optimum sits strictly between 'block all' and 'block none'."""
    y, p, amt = make_case()
    cm = CostModel()
    best_t, table = cm.optimal_threshold(y, p, amt)
    best = table["net_cost"].min()
    assert best <= cm.expected_cost(y, p, amt, 1.01)["net_cost"]
    assert best <= cm.expected_cost(y, p, amt, 0.0)["net_cost"]
    assert 0.0 < best_t < 1.0


def test_higher_false_positive_cost_raises_the_threshold():
    """If false alarms get more expensive, be more conservative.

    This is the economic behaviour the whole risk engine exists to express.
    """
    y, p, amt = make_case()
    cheap, _ = CostModel(fp_cost=1.0).optimal_threshold(y, p, amt)
    dear, _ = CostModel(fp_cost=200.0).optimal_threshold(y, p, amt)
    assert dear > cheap


# ------------------------------------------------------------- metrics ----
def test_pr_auc_lift_is_relative_to_the_base_rate():
    """PR-AUC alone is meaningless; the lift is what carries information."""
    y, p, _ = make_case()
    m = compute_metrics(y, p)
    assert m.base_rate == pytest.approx(float(y.mean()))
    assert m.pr_auc_lift == pytest.approx(m.pr_auc / m.base_rate)
    assert m.pr_auc > m.base_rate       # better than random


def test_random_scores_give_pr_auc_near_the_base_rate():
    """The definition of 'no better than random'."""
    n = 20_000
    y = (RNG.random(n) < 0.035).astype(int)
    p = RNG.random(n)                    # pure noise
    m = compute_metrics(y, p)
    assert m.pr_auc == pytest.approx(m.base_rate, abs=0.02)
    assert m.pr_auc_lift == pytest.approx(1.0, abs=0.6)


# ---------------------------------------------------------- thresholds ----
def test_alert_budget_flags_the_requested_share():
    y, p, _ = make_case()
    for budget in (0.005, 0.01, 0.05):
        t = threshold_for_alert_budget(p, budget)
        assert confusion_at(y, p, t)["alert_rate"] == pytest.approx(budget, abs=0.003)


def test_precision_target_is_met_when_reachable():
    y, p, _ = make_case()
    t = threshold_for_precision(y, p, 0.5)
    assert confusion_at(y, p, t)["precision"] >= 0.5


def test_raising_the_threshold_trades_recall_for_precision():
    """The fundamental trade-off, asserted rather than assumed."""
    y, p, _ = make_case()
    low = confusion_at(y, p, 0.2)
    high = confusion_at(y, p, 0.6)
    assert high["precision"] >= low["precision"]
    assert high["recall"] <= low["recall"]
    assert high["alert_rate"] <= low["alert_rate"]


# ------------------------------------------------- narrative formatting ----
def test_fmt_handles_numpy_nan_not_just_python_nan():
    """Regression: numpy float32 is NOT a Python float.

    The first version tested `isinstance(v, float)` before `isnan`, so numpy
    NaN fell through to `str(v)` and narratives read
    "a count of related addresses = nan" - nonsense to an analyst.
    """
    from risklens.genai.narratives import _fmt

    assert _fmt(np.float32("nan")) == "not recorded"
    assert _fmt(np.float64("nan")) == "not recorded"
    assert _fmt(float("nan")) == "not recorded"
    assert _fmt(None) == "not recorded"


def test_fmt_renders_real_values_readably():
    from risklens.genai.narratives import _fmt

    assert _fmt(np.int32(7)) == "7"
    assert _fmt(np.float32(3.14159)) == "3.14"
    assert _fmt("visa") == "visa"
    assert _fmt(np.bool_(True)) == "yes"


def test_band_actions_match_the_policy_table():
    """Regression: the copilot once reported CRITICAL as the HIGH action.

    CRITICAL must be decline-and-contact, not hold-and-review. This is a
    deterministic lookup precisely so a language model cannot shift a row.
    """
    from risklens.genai.narratives import BAND_ACTIONS, BAND_OWNER, band_risk

    assert "decline" in BAND_ACTIONS["CRITICAL"]
    assert "hold" in BAND_ACTIONS["HIGH"]
    assert BAND_ACTIONS["CRITICAL"] != BAND_ACTIONS["HIGH"]
    assert set(BAND_ACTIONS) == set(BAND_OWNER)
    # every band the risk function can emit must have an action
    for p in (0.99, 0.80, 0.50, 0.20, 0.01):
        assert band_risk(p) in BAND_ACTIONS
