"""Stage 2 - statistical testing.

Purpose
-------
EDA charts show you a difference. Statistics tell you whether that difference
is real or is what randomness looks like. With 590k rows, a third question
becomes the important one: an effect can be overwhelmingly *significant* and
still be far too small to matter.

So every test here returns THREE things:
    statistic  - the test value
    p_value    - is the difference real?
    effect     - is the difference big enough to care about?

Why the effect size matters so much at this scale
-------------------------------------------------
p-values shrink as n grows. At n=590,000 almost any difference is
"significant" at p<0.05. A 0.01% difference in mean transaction amount will
be significant and useless. Reporting p-values alone at this sample size is a
classic mistake, and an interviewer may well probe it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestResult:
    """One statistical test: what was tested, is it real, is it big."""

    feature: str
    test: str
    statistic: float
    p_value: float
    effect_name: str
    effect_size: float
    n: int
    interpretation: str

    def as_row(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "test": self.test,
            "statistic": round(self.statistic, 4),
            "p_value": self.p_value,
            "effect": self.effect_name,
            "effect_size": round(self.effect_size, 4),
            "n": self.n,
            "interpretation": self.interpretation,
        }


# ------------------------------------------------------------ categorical ---
def cramers_v(confusion: np.ndarray) -> float:
    """Cramer's V - association strength between two categoricals, in [0, 1].

    Chi-square alone cannot be compared across features because it scales with
    n. Cramer's V normalises it:

        V = sqrt( chi2 / (n * min(rows-1, cols-1)) )

    Rule of thumb: 0.1 weak, 0.3 moderate, 0.5 strong.
    """
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum()
    min_dim = min(confusion.shape[0] - 1, confusion.shape[1] - 1)
    if n == 0 or min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def chi_square_test(
    df: pd.DataFrame, column: str, *, target: str, min_count: int = 50
) -> TestResult | None:
    """Is this categorical feature associated with fraud?

    H0: the category and the class label are independent.

    Rare categories are pooled into "__other__" first. Chi-square is unreliable
    when expected cell counts fall below ~5, and this dataset has categoricals
    with hundreds of rare levels.
    """
    s = df[column].astype("object").where(df[column].notna(), "__missing__")
    counts = s.value_counts()
    keep = counts[counts >= min_count].index
    s = s.where(s.isin(keep), "__other__")

    confusion = pd.crosstab(s, df[target])
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return None

    chi2, p, _, _ = stats.chi2_contingency(confusion.to_numpy(), correction=False)
    v = cramers_v(confusion.to_numpy())

    if p >= 0.05:
        interp = "no detectable association"
    elif v < 0.1:
        interp = "significant but negligible - large n, tiny effect"
    elif v < 0.3:
        interp = "weak but usable association"
    else:
        interp = "moderate-to-strong association"

    return TestResult(
        feature=column,
        test="chi-square",
        statistic=float(chi2),
        p_value=float(p),
        effect_name="cramers_v",
        effect_size=v,
        n=int(confusion.to_numpy().sum()),
        interpretation=interp,
    )


# ---------------------------------------------------------------- numeric ---
def cliffs_delta_from_u(u: float, n1: int, n2: int) -> float:
    """Cliff's delta in [-1, 1], derived from the Mann-Whitney U statistic.

        delta = 2U / (n1 * n2) - 1

    Interpretation: the probability a random fraud value exceeds a random
    legitimate value, rescaled. 0 means the distributions overlap completely.

    Rule of thumb: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium.
    """
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(2.0 * u / (n1 * n2) - 1.0)


def mannwhitney_test(
    df: pd.DataFrame, column: str, *, target: str
) -> TestResult | None:
    """Do fraud and legitimate transactions differ on this numeric feature?

    Why Mann-Whitney U and not a t-test
    -----------------------------------
    A t-test compares MEANS and assumes roughly normal distributions.
    TransactionAmt is heavily right-skewed with extreme outliers, so its mean
    is not a stable description of the data.

    Mann-Whitney U is non-parametric: it ranks all values and asks whether one
    group tends to rank higher. It makes no distributional assumption and is
    robust to outliers - both of which matter for money.
    """
    a = df.loc[df[target] == 1, column].dropna()
    b = df.loc[df[target] == 0, column].dropna()
    if len(a) < 20 or len(b) < 20:
        return None

    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = cliffs_delta_from_u(float(u), len(a), len(b))

    ad = abs(d)
    if p >= 0.05:
        interp = "no detectable difference"
    elif ad < 0.147:
        interp = "significant but negligible - large n, tiny effect"
    elif ad < 0.33:
        interp = "small but real difference"
    elif ad < 0.474:
        interp = "medium difference"
    else:
        interp = "large difference"

    return TestResult(
        feature=column,
        test="mann-whitney-u",
        statistic=float(u),
        p_value=float(p),
        effect_name="cliffs_delta",
        effect_size=d,
        n=int(len(a) + len(b)),
        interpretation=interp,
    )


# ------------------------------------------------------------------ drift ---
def population_stability_index(
    expected: pd.Series, actual: pd.Series, *, bins: int = 10
) -> float:
    """PSI - the standard drift metric in credit risk and banking.

        PSI = sum over bins of  (a_i - e_i) * ln(a_i / e_i)

    where e_i and a_i are the proportion of the expected (reference/training)
    and actual (new/production) populations falling in bin i.

    Industry thresholds:
        PSI < 0.10  no significant shift
        0.10-0.25   moderate shift - investigate
        PSI > 0.25  major shift - the model likely needs retraining

    This is exactly the metric a model-risk team at a bank would ask for, and
    it is why we care whether the training and test periods look alike.
    Quantile bins come from the EXPECTED distribution only - deriving them
    from the actual would hide the very drift we are trying to measure.
    """
    e = expected.dropna()
    a = actual.dropna()
    if len(e) == 0 or len(a) == 0:
        return float("nan")

    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_pct = np.histogram(e, bins=edges)[0] / len(e)
    a_pct = np.histogram(a, bins=edges)[0] / len(a)

    # Laplace-style floor: a zero in either array makes the log term infinite.
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], *, bins: int = 10
) -> pd.DataFrame:
    """PSI for many columns at once - e.g. train vs test partition.

    A high-PSI feature is one whose distribution moved between periods. Those
    are the features most likely to degrade in production, and the ones worth
    monitoring after deployment.
    """
    rows = []
    for col in columns:
        if col not in reference.columns or col not in current.columns:
            continue
        if not pd.api.types.is_numeric_dtype(reference[col]):
            continue
        psi = population_stability_index(reference[col], current[col], bins=bins)
        if np.isnan(psi):
            continue
        band = (
            "stable" if psi < 0.10
            else "moderate shift" if psi < 0.25
            else "MAJOR SHIFT"
        )
        rows.append({"column": col, "psi": round(psi, 4), "verdict": band})
    return (
        pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    )


def run_test_battery(
    df: pd.DataFrame,
    *,
    target: str,
    categorical: list[str],
    numeric: list[str],
) -> pd.DataFrame:
    """Run every applicable test and return one ranked table.

    Ranked by effect size, NOT by p-value - see the module docstring for why
    p-values are close to meaningless for ranking at this sample size.
    """
    results: list[TestResult] = []
    for col in categorical:
        if col in df.columns:
            r = chi_square_test(df, col, target=target)
            if r:
                results.append(r)
    for col in numeric:
        if col in df.columns:
            r = mannwhitney_test(df, col, target=target)
            if r:
                results.append(r)

    if not results:
        return pd.DataFrame()
    return (
        pd.DataFrame([r.as_row() for r in results])
        .assign(abs_effect=lambda d: d["effect_size"].abs())
        .sort_values("abs_effect", ascending=False)
        .drop(columns="abs_effect")
        .reset_index(drop=True)
    )
