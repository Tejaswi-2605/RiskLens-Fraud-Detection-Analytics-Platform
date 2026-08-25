"""Stage 4 - supervised modelling.

Two models, deliberately:

  1. Logistic Regression  - the BASELINE. Simple, linear, fast, calibrated by
     construction. Its job is not to win; its job is to establish the number
     that a complex model must beat to justify its complexity.

  2. XGBoost              - the CANDIDATE. Gradient-boosted trees, which are
     the standard for tabular fraud problems.

If XGBoost cannot clearly beat logistic regression, the honest answer is to
ship logistic regression. A model you can explain to a regulator has real
value in a bank; complexity you cannot justify does not.

Leakage discipline in this file
-------------------------------
Every fitted step (imputer, scaler, frequency encoder) lives INSIDE an
sklearn Pipeline. `pipe.fit(X_train, y_train)` therefore cannot see
validation or test data - not because I remembered to be careful, but
because the object graph makes it impossible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger(__name__)


# =========================================================================
# Class imbalance
# =========================================================================
def scale_pos_weight(y: pd.Series) -> float:
    """The imbalance ratio: negatives / positives.

    With 3.5% fraud this is about 27.6, meaning "treat one missed fraud as
    27.6 times worse than one false alarm during training."

    Why this rather than SMOTE
    --------------------------
    SMOTE invents synthetic minority rows by interpolating between real ones.
    Three problems here:

      * Interpolating between two frauds committed by different people, with
        different methods, produces a transaction that never existed and
        could not exist.
      * It is expensive on 438k x ~450 data.
      * It changes the base rate, so the predicted probabilities are no
        longer calibrated to reality - and Stage 5 needs calibrated
        probabilities to compute expected loss in pounds.

    Reweighting achieves the same rebalancing by changing the LOSS FUNCTION
    rather than the data. Nothing is invented. It is also one line.
    """
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return float(neg / max(pos, 1))


# =========================================================================
# Baseline: logistic regression
# =========================================================================
def build_baseline_pipeline(
    numeric_cols: list[str], categorical_cols: list[str]
) -> Pipeline:
    """Logistic regression with all preprocessing INSIDE the pipeline.

    Why each step exists
    --------------------
    median imputation  - LogReg cannot accept NaN at all. Median rather than
                         mean because our numerics are heavily skewed; the
                         mean of a skewed column sits where no real value is.
                         Note we have ALREADY added explicit `_isna` flags in
                         Stage 3, so imputing does not destroy the
                         missingness information - the flag preserves it.

    StandardScaler     - LogReg with L2 regularisation penalises large
                         coefficients. Without scaling, a feature measured in
                         thousands gets a tiny coefficient and is effectively
                         penalised more than one measured in units. Scaling
                         makes the penalty fair across features.

    OneHotEncoder      - LogReg needs numbers. `min_frequency=0.01` pools
                         rare categories into one "infrequent" column, which
                         prevents thousands of near-empty columns and reduces
                         overfitting to categories seen a handful of times.
                         `handle_unknown="infrequent_if_exist"` means a
                         category appearing only at scoring time does not
                         crash the API in Stage 10.

    class_weight="balanced" - the LogReg equivalent of scale_pos_weight.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.01,
            sparse_output=True,
        )),
    ])
    pre = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])


# =========================================================================
# Candidate: XGBoost
# =========================================================================
def build_xgboost(y_train: pd.Series, **overrides: Any):
    """Gradient-boosted trees, configured for imbalanced tabular fraud.

    Why XGBoost for this problem
    ----------------------------
    * Handles NaN natively - it LEARNS which side of a split missing values
      belong on. Given that 229 of our 434 columns are >50% missing, this is
      not a convenience, it is the single biggest reason to prefer trees here.
    * Handles categorical features natively (`enable_categorical=True`), so
      no one-hot explosion on 17,000 card1 values.
    * Captures interactions automatically. Fraud is conjunctive - "new device
      AND unusual hour AND rare card" - which is exactly what a tree path is.
    * Scale-invariant. No normalisation needed.

    Key hyperparameters, and why these values
    -----------------------------------------
    n_estimators=600 + early stopping
        Boosting adds trees one at a time, each correcting the previous
        errors. Too many overfits. Rather than guess the number, we allow
        600 and let early stopping pick the point where validation PR-AUC
        stops improving.

    learning_rate=0.05
        How much each new tree contributes. Lower = slower but more accurate.
        0.05 with ~600 trees is a standard, safe trade.

    max_depth=6
        Depth 6 means a tree can express interactions of up to 6 features.
        Deeper memorises; shallower cannot capture conjunctive fraud patterns.

    subsample / colsample_bytree = 0.8
        Each tree sees 80% of rows and 80% of columns. This is bagging inside
        boosting: it decorrelates the trees and is a strong regulariser.

    min_child_weight=5
        A leaf must carry at least this much weight. Prevents a leaf built
        from 2 fraud cases, which is memorisation, not learning.

    eval_metric="aucpr"
        Optimise PR-AUC, NOT accuracy or ROC-AUC. With 3.5% positives, PR-AUC
        is the metric that actually reflects performance on the rare class.
    """
    from xgboost import XGBClassifier

    params: dict[str, Any] = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight(y_train),
        eval_metric="aucpr",
        early_stopping_rounds=50,
        enable_categorical=True,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    params.update(overrides)
    return XGBClassifier(**params)


# =========================================================================
# Column typing helper
# =========================================================================
@dataclass
class ColumnSpec:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.numeric) + len(self.categorical)


def split_column_types(df: pd.DataFrame, features: list[str]) -> ColumnSpec:
    """Partition feature columns into numeric vs categorical."""
    numeric, categorical = [], []
    for c in features:
        if c not in df.columns:
            continue
        if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object:
            categorical.append(c)
        else:
            numeric.append(c)
    return ColumnSpec(numeric=numeric, categorical=categorical)
