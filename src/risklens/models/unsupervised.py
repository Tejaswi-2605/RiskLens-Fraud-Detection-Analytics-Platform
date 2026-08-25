"""Stage 6 - unsupervised learning: anomaly detection and fraud typologies.

Why add unsupervised methods when we already have a supervised model
--------------------------------------------------------------------
The supervised model can only recognise fraud that RESEMBLES fraud it was
trained on. That is a real weakness in an adversarial domain: criminals
change methods precisely to avoid known patterns.

Two distinct jobs here, often confused:

  ANOMALY DETECTION  finds transactions unlike ANYTHING seen before,
                     including fraud types that did not exist in training.
                     It is a safety net for the unknown-unknowns.

  CLUSTERING         groups the KNOWN fraud into behavioural families
                     ("typologies"). It does not detect anything - it
                     describes and organises what we already caught.

The second is what lets an analyst say "this is a card-testing attack" rather
than "the model scored 0.87", and it is what gives the Stage 9 copilot
something meaningful to name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# =========================================================================
# Anomaly detection
# =========================================================================
def fit_isolation_forest(
    X: pd.DataFrame, *, contamination: float = 0.035, random_state: int = 42
):
    """Isolation Forest - unsupervised outlier detection.

    How it works, intuitively
    -------------------------
    Build random trees by repeatedly picking a random feature and a random
    split point. Then ask: how many splits does it take to ISOLATE this row
    into a leaf of its own?

    Tiny example. Ages: 25, 26, 27, 28, 29, 95.
      To isolate 95, one random split near 60 does it        -> depth 1
      To isolate 27, you need several splits to separate it
      from 26 and 28                                          -> depth 4-5

    Outliers are isolated QUICKLY because they sit in sparse regions. So
    short average path length = anomalous. That is the whole algorithm.

    Why this rather than a one-class SVM or a local-density method
    --------------------------------------------------------------
      * Linear time, so it handles 438k rows; one-class SVM is roughly
        quadratic and would not finish.
      * No distance metric needed, so no scaling required and no curse of
        dimensionality across 500+ features.
      * Handles mixed feature scales natively.

    IMPORTANT: it is fitted WITHOUT labels. It never sees `isFraud`. That is
    the point - it must be able to flag a fraud type nobody has labelled yet.

    `contamination` is our prior guess at the outlier rate. We set it to the
    known fraud rate so the flagged volume is operationally comparable, but
    the model is not using the labels to get there.
    """
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples=min(256_000, len(X)),
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    log.info("IsolationForest fitted on %s rows", f"{len(X):,}")
    return model


def anomaly_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous.

    sklearn's `score_samples` returns higher values for NORMAL points, which
    is the opposite of what everyone expects. We negate it so that "big
    number = suspicious", matching the mental model of every other score in
    this project.
    """
    return -model.score_samples(X)


def evaluate_anomaly_detector(
    scores: np.ndarray, y_true: np.ndarray, base_rate: float
) -> dict[str, Any]:
    """How much fraud does a purely unsupervised detector find?

    Expectation management: it will be MUCH worse than the supervised model,
    and that is not a failure. The supervised model had 15,364 labelled
    examples; this had none. The interesting question is whether it beats
    random, because that would mean fraud genuinely is anomalous in feature
    space - which justifies keeping it as a safety net for novel attacks.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    pr = float(average_precision_score(y_true, scores))
    return {
        "pr_auc": round(pr, 5),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 5),
        "lift_over_random": round(pr / base_rate, 2) if base_rate else None,
        "note": (
            "Unsupervised - never saw a label. Compare against the base rate, "
            "not against the supervised model."
        ),
    }


# =========================================================================
# Fraud typologies via clustering
# =========================================================================
@dataclass
class Typology:
    """One behavioural family of fraud, described in business language."""

    cluster_id: int
    n_cases: int
    share: float
    avg_amount: float
    label: str
    signature: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "n_cases": self.n_cases,
            "share": round(self.share, 4),
            "avg_amount": round(self.avg_amount, 2),
            "label": self.label,
            "signature": {k: round(v, 3) for k, v in self.signature.items()},
        }


def cluster_fraud_typologies(
    fraud_rows: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_clusters: int = 5,
    random_state: int = 42,
):
    """Group CONFIRMED fraud cases into behavioural families.

    Note we cluster ONLY the fraud rows, not the whole dataset. Clustering
    everything would just rediscover the majority class; we already know
    which rows are fraud and are asking a different question: what KINDS of
    fraud are there?

    Why K-Means
    -----------
    Fast, scales to our size, and produces centroids that are directly
    interpretable as "the average member of this family".

    Its weaknesses matter less here: it assumes roughly spherical clusters of
    similar size, which is a poor fit for arbitrary shapes - but we want
    coarse, nameable groups for analysts, not a precise density partition.

    Why scaling is REQUIRED here (unlike for trees)
    -----------------------------------------------
    K-Means minimises Euclidean distance. Without scaling, a feature ranging
    0-31,000 (amount) completely dominates one ranging 0-1 (is_night), so the
    clustering would effectively be "amount buckets".

    Why we choose k with the silhouette score
    -----------------------------------------
    Silhouette measures how similar a point is to its own cluster versus the
    nearest other cluster, from -1 to +1. Around 0 means the clusters
    overlap; nearer 1 means they are well separated. It gives a principled
    way to pick k rather than guessing.
    """
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = fraud_rows[feature_cols]
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    Xs = pipe.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)
    log.info("clustered %s fraud cases into %d typologies", f"{len(X):,}", n_clusters)
    return km, pipe, labels


def describe_typologies(
    fraud_rows: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    *,
    amount_col: str = "TransactionAmt",
) -> list[Typology]:
    """Turn anonymous cluster IDs into named, business-readable typologies.

    A cluster called "3" is useless to a fraud analyst. We compute how each
    cluster differs from the fraud average (in standard deviations) and use
    the strongest deviations to auto-generate a descriptive label.

    This is what makes clustering ACTIONABLE rather than decorative, and it
    is what the Stage 9 copilot cites when explaining an alert.
    """
    df = fraud_rows.copy()
    df["_cluster"] = labels
    overall_mean = df[feature_cols].mean()
    overall_std = df[feature_cols].std().replace(0, np.nan)

    out: list[Typology] = []
    for cid in sorted(df["_cluster"].unique()):
        sub = df[df["_cluster"] == cid]
        # z-score of this cluster's mean vs the overall fraud mean
        z = ((sub[feature_cols].mean() - overall_mean) / overall_std).dropna()
        top = z.reindex(z.abs().sort_values(ascending=False).index).head(4)

        parts = []
        for feat, val in top.items():
            parts.append(f"{'high' if val > 0 else 'low'} {feat}")
        label = ", ".join(parts) if parts else "undifferentiated"

        out.append(Typology(
            cluster_id=int(cid),
            n_cases=int(len(sub)),
            share=float(len(sub) / len(df)),
            avg_amount=float(sub[amount_col].mean()) if amount_col in sub else float("nan"),
            label=label,
            signature=top.to_dict(),
        ))
    return out


def choose_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> pd.DataFrame:
    """Silhouette score across candidate k, to pick the number of clusters.

    Computed on a subsample: silhouette is O(n^2) in memory because it needs
    pairwise distances, so running it on 15,000 fraud rows directly would
    allocate a 15,000 x 15,000 matrix.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_scaled), size=min(4_000, len(X_scaled)), replace=False)
    Xs = X_scaled[idx]

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(Xs)
        rows.append({
            "k": k,
            "silhouette": round(float(silhouette_score(Xs, lab)), 4),
            "inertia": round(float(km.inertia_), 1),
        })
    return pd.DataFrame(rows)
