"""Tests for probability calibration.

Two lessons are encoded here.

1. A calibrator must never REORDER predictions. The correct test for that is
   counting inversions, not comparing PR-AUC - the first version of the
   calibration script checked PR-AUC equality and reported a spurious failure
   on a perfectly correct isotonic calibrator, because isotonic creates ties
   and `average_precision` treats tied scores differently.

2. Selection between isotonic and Platt is not purely about Brier. Isotonic
   usually wins slightly, but it wins by collapsing the score into a handful
   of steps, which destroys the ability to place a threshold at an arbitrary
   alert budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from risklens.models.calibrate import (
    ProbabilityCalibrator,
    count_inversions,
    expected_calibration_error,
    fit_and_compare,
    reliability_table,
)

RNG = np.random.default_rng(11)


def make_overconfident(n: int = 8_000, rate: float = 0.035):
    """Scores that rank well but are systematically too confident.

    This mimics what `scale_pos_weight` does: the model is trained as though
    the classes were balanced, so it pushes probabilities toward the extremes.
    """
    y = (RNG.random(n) < rate).astype(int)
    latent = RNG.beta(2, 6, n) + y * RNG.uniform(0.2, 0.5, n)
    # squash-then-stretch: preserves order, inflates confidence
    p = np.clip(latent**0.45, 1e-6, 1 - 1e-6)
    return y, p


# ------------------------------------------------------------ monotonic ---
def test_calibration_never_reorders_predictions():
    """The core guarantee. Equal is fine; lower is a real inversion."""
    y, p = make_overconfident()
    for method in ("isotonic", "sigmoid"):
        c = ProbabilityCalibrator(method).fit(p[:4000], y[:4000])
        assert count_inversions(p, c.transform(p)) == 0


def test_platt_is_strictly_monotonic_but_isotonic_creates_ties():
    """The trade-off that drives method selection.

    Isotonic collapses many distinct scores into one, which is exactly why it
    can beat Platt on calibration and still be the worse choice.
    """
    y, p = make_overconfident()
    iso = ProbabilityCalibrator("isotonic").fit(p[:4000], y[:4000]).transform(p)
    plt = ProbabilityCalibrator("sigmoid").fit(p[:4000], y[:4000]).transform(p)

    assert len(np.unique(plt)) == pytest.approx(len(np.unique(p)), rel=0.02)
    assert len(np.unique(iso)) < len(np.unique(p)) / 10


def test_count_inversions_detects_a_real_reordering():
    """Guard the guard: the check must actually fire when order breaks."""
    raw = np.array([0.1, 0.2, 0.3, 0.4])
    good = np.array([0.05, 0.10, 0.10, 0.30])   # non-decreasing, ties allowed
    bad = np.array([0.05, 0.30, 0.10, 0.40])    # 0.30 then 0.10 - reordered
    assert count_inversions(raw, good) == 0
    assert count_inversions(raw, bad) == 1


# ---------------------------------------------------------- calibration ---
def test_calibration_improves_brier_and_ece():
    y, p = make_overconfident()
    cal, result, stats = fit_and_compare(p[:4000], y[:4000], p[4000:], y[4000:])
    assert result.brier_after < result.brier_before
    assert result.ece_after < result.ece_before
    assert result.improvement_pct > 0


def test_calibration_removes_overconfidence():
    """The observable symptom: the maximum predicted probability comes down."""
    y, p = make_overconfident()
    _, result, _ = fit_and_compare(p[:4000], y[:4000], p[4000:], y[4000:])
    assert result.max_prob_after <= result.max_prob_before


def test_selection_prefers_platt_when_isotonic_barely_wins():
    """Encodes the judgement call, so it cannot drift silently."""
    y, p = make_overconfident()
    _, result, stats = fit_and_compare(
        p[:4000], y[:4000], p[4000:], y[4000:], tie_penalty_margin=0.99
    )
    # with an impossible margin, isotonic can never clear it
    assert result.method == "sigmoid"
    assert stats["_selection"]["chosen"] == "sigmoid"


# ------------------------------------------------------------- metrics ----
def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    """Sanity anchor: if predicted == observed everywhere, ECE is ~0."""
    n = 20_000
    p = RNG.uniform(0.01, 0.6, n)
    y = (RNG.random(n) < p).astype(int)   # outcomes drawn AT the stated rate
    assert expected_calibration_error(y, p) < 0.02


def test_ece_is_large_for_a_badly_calibrated_predictor():
    n = 20_000
    y = (RNG.random(n) < 0.05).astype(int)
    p = np.full(n, 0.90)                  # claims 90% when truth is 5%
    assert expected_calibration_error(y, p) > 0.5


def test_reliability_table_bins_sum_to_the_population():
    y, p = make_overconfident()
    rows = reliability_table(y, p, n_bins=10)
    assert sum(r["n"] for r in rows) == len(y)
    for r in rows:
        assert 0.0 <= r["mean_predicted"] <= 1.0
        assert 0.0 <= r["observed_rate"] <= 1.0
