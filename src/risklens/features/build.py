"""Stage 3 (part 2) - feature engineering.

The organising principle of this file
-------------------------------------
Features come in two kinds, and confusing them is how leakage happens:

  DETERMINISTIC (row-wise)   log(amount), hour-of-day, is-this-value-missing
      Computed from ONE row. Nothing is learned from other rows. Safe to
      apply anywhere, any time, in any order.

  FITTED (cross-row)         frequency encoding, target encoding, scaling
      LEARN a parameter by looking across many rows. Must be fitted on the
      TRAINING partition only and then merely APPLIED to val/test.

Everything in `add_deterministic_features` is the first kind.
`FrequencyEncoder` is the second kind and is a proper scikit-learn
transformer, so it can live inside a Pipeline where the fit/transform
discipline is enforced by the framework rather than by my memory.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400

# Columns whose ABSENCE proved highly predictive in Stage 2 EDA.
# Fraud rate was ~10.3% when id_04 was present vs ~2.6% when missing.
MISSING_INDICATOR_COLS = [
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_09",
    "id_13", "id_31", "DeviceType", "DeviceInfo",
    "dist1", "dist2", "D1", "D15", "addr1", "P_emaildomain",
]

# High-cardinality identifiers worth frequency-encoding.
FREQ_ENCODE_COLS = [
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "DeviceInfo", "id_31", "id_30",
]


# =========================================================================
# DETERMINISTIC FEATURES - safe, row-wise, no fitting
# =========================================================================
def add_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features. Nothing here learns from other rows.

    Every feature below can be computed for a SINGLE transaction arriving at
    the API in Stage 10, with no access to any dataset. That is the test for
    whether a feature is deterministic - and it is also what makes
    training/serving skew impossible for these columns.
    """
    out = df.copy()

    # ---- amount transformations -----------------------------------------
    # log1p, not log: log(0) is -inf, log1p(0) is 0. Amounts can be tiny.
    # Rationale: money is heavily right-skewed. The log compresses the long
    # tail so a £30,000 transaction stops dominating distance-based and
    # linear models. Trees do not need it, but the baseline LogReg does.
    out["amt_log"] = np.log1p(out["TransactionAmt"])

    # The cents portion. Genuine retail prices cluster at .00, .99, .95.
    # Stolen-card testing often produces unusual decimals, and currency
    # conversion produces long ones. This is a well-known IEEE-CIS signal.
    cents = out["TransactionAmt"] - np.floor(out["TransactionAmt"])
    out["amt_cents"] = cents.round(4)
    out["amt_is_round"] = (cents == 0).astype("int8")

    # ---- time-of-day cycle ----------------------------------------------
    # TransactionDT is seconds from an UNKNOWN origin, so we cannot recover a
    # calendar date. But the origin is CONSTANT, so modulo arithmetic gives a
    # valid RELATIVE hour and weekday: the labels are shifted, the cycle is
    # real. Stage 2 confirmed a genuine daily pattern.
    out["hour"] = ((out["TransactionDT"] // 3600) % 24).astype("int8")
    out["dayofweek"] = ((out["TransactionDT"] // SECONDS_PER_DAY) % 7).astype("int8")
    out["is_night"] = out["hour"].between(0, 6).astype("int8")

    # ---- missingness as an explicit feature -----------------------------
    # Stage 2's headline finding: whether identity data exists is one of the
    # strongest signals in the dataset (10.31% vs 2.61% fraud rate). Trees
    # can split on NaN, but making it explicit lets a linear model use it too
    # and makes the effect visible in SHAP.
    for col in MISSING_INDICATOR_COLS:
        if col in out.columns:
            out[f"{col}_isna"] = out[col].isna().astype("int8")

    # How much of this row is missing overall. A compact summary of the
    # 14 correlated V-blocks found in Stage 2.
    out["n_missing"] = df.isna().sum(axis=1).astype("int16")

    # ---- email domain simplification ------------------------------------
    # 'gmail.com' and 'gmail' are the same provider; 'live.com.mx' tells us
    # both provider and region. Splitting reduces cardinality without
    # discarding information.
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in out.columns:
            s = out[col].astype("object")
            out[f"{col}_provider"] = (
                s.str.split(".").str[0].fillna("__missing__").astype("category")
            )
            out[f"{col}_suffix"] = (
                s.str.split(".").str[-1].fillna("__missing__").astype("category")
            )
    # Do the sender and receiver domains match? A mismatch is a classic
    # account-takeover indicator.
    if {"P_emaildomain", "R_emaildomain"}.issubset(out.columns):
        p = out["P_emaildomain"].astype("object")
        r = out["R_emaildomain"].astype("object")
        out["email_domains_match"] = (
            (p == r).where(p.notna() & r.notna(), np.nan).astype("float32")
        )

    log.info(
        "deterministic features: %d -> %d columns (+%d)",
        df.shape[1], out.shape[1], out.shape[1] - df.shape[1],
    )
    return out


# =========================================================================
# FITTED TRANSFORMER - learns from train, applies to everything
# =========================================================================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace a category with HOW OFTEN it appeared in the training data.

    What it does
    ------------
    `card1 = 13926` becomes `card1_freq = 0.00042` (it was 0.042% of training
    rows). The raw ID is meaningless as a number; its rarity is not.

    Why it helps for fraud
    ----------------------
    Rare is suspicious. A card ID seen once in 400,000 transactions behaves
    very differently from one seen 5,000 times. Frequency encoding turns a
    high-cardinality identifier (17,000 distinct card1 values) into ONE
    informative numeric column, instead of 17,000 one-hot columns.

    Why it is a FITTED transformer and not a helper function
    --------------------------------------------------------
    The counts must come from the TRAINING data only. If you computed them
    over train+test you would leak: a card's test-period frequency tells the
    model something about the future. Making this a scikit-learn transformer
    means `fit` runs on train inside a Pipeline, and val/test only ever see
    `transform`. The framework enforces the discipline, not my memory.

    Unseen categories
    -----------------
    A card that appears only in test was never in the training counts. It
    maps to `unseen_value` (default 0), which is meaningful here rather than
    arbitrary: "never observed during training" IS the rarest possible case,
    and rarity is what this feature encodes.
    """

    def __init__(self, columns: list[str] | None = None, unseen_value: float = 0.0):
        self.columns = columns
        self.unseen_value = unseen_value

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        cols = self.columns or []
        self.columns_ = [c for c in cols if c in X.columns]
        # normalize=True gives a proportion, not a raw count, so the feature
        # does not change scale when the training set size changes.
        self.freq_maps_: dict[str, pd.Series] = {
            c: X[c].astype("object").value_counts(normalize=True, dropna=True)
            for c in self.columns_
        }
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.columns_:
            out[f"{c}_freq"] = (
                out[c].astype("object")
                .map(self.freq_maps_[c])
                .fillna(self.unseen_value)
                .astype("float32")
            )
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_freq" for c in self.columns_], dtype=object)


def select_model_features(
    df: pd.DataFrame, *, target: str, drop: list[str] | None = None
) -> list[str]:
    """Choose the columns the model may see.

    Two exclusions matter, and both are leakage controls:

      TransactionID  - a meaningless surrogate key. It also happens to
                       increase with time, so a tree could use it to identify
                       WHICH PERIOD a row came from. That is leakage through
                       the index.

      TransactionDT  - the raw timestamp, for exactly the same reason. It
                       monotonically separates train from test, so a tree
                       would learn "large DT means test set". We keep its
                       DERIVED cyclical parts (hour, dayofweek) which carry
                       behaviour rather than position in time.
    """
    always_drop = {target, "TransactionID", "TransactionDT", *(drop or [])}
    return [c for c in df.columns if c not in always_drop]
