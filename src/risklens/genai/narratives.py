"""Stage 8 (part 1) - turning a scored transaction into text.

Why a fraud project needs an NLP component at all
--------------------------------------------------
The IEEE-CIS dataset contains no free text. So rather than inventing a fake
text column, we GENERATE the text that a real fraud operation actually
produces: the **case narrative** an analyst writes when investigating an alert.

That gives us a genuine, non-contrived NLP corpus:

    structured transaction + SHAP reason codes  ->  case narrative (text)
                                                ->  embeddings
                                                ->  semantic search
                                                       "have we seen this before?"

This is the honest framing. We are not bolting a chatbot onto a classifier; we
are building the artefact that a fraud analyst produces and then making it
searchable - which is exactly what a real case-management system does.

Important honesty note
----------------------
These narratives are DERIVED from real transaction data via a deterministic
template. They are not real analyst write-ups, and the module labels them as
machine-generated. Never present them as human case notes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Plain-English names for the dataset's cryptic columns. An analyst cannot act
# on "V257 = 3.2"; the whole point of a narrative is to remove that barrier.
FEATURE_GLOSSARY: dict[str, str] = {
    "amt_log": "transaction amount",
    "TransactionAmt": "transaction amount",
    "amt_cents": "the cents portion of the amount",
    "amt_is_round": "a round-number amount",
    "hour": "the hour of day",
    "is_night": "an overnight transaction",
    "dayofweek": "the day of week",
    "n_missing": "the amount of missing data on this record",
    "ProductCD": "the product type",
    "card4": "the card network",
    "card6": "the card type (debit/credit)",
    "card1_freq": "how common this card identifier is",
    "card2_freq": "how common this card group is",
    "addr1_freq": "how common this billing region is",
    "P_emaildomain": "the payer email domain",
    "P_emaildomain_freq": "how common the payer email domain is",
    "R_emaildomain": "the recipient email domain",
    "email_domains_match": "whether payer and recipient email domains match",
    "DeviceType": "the device type",
    "DeviceInfo_freq": "how common this device is",
    "id_31": "the browser",
    "id_31_freq": "how common this browser is",
    "dist1": "the distance between billing and transaction location",
    "dist2": "a secondary distance measure",
    "D15": "days since a prior related transaction",
    "D1": "days since the card was first seen",
    "C1": "a count of related addresses",
    "C13": "a count of related payment activity",
    "C14": "a count of distinct related entities",
}


def humanise(feature: str) -> str:
    """Map a raw column name to something an analyst can read."""
    if feature in FEATURE_GLOSSARY:
        return FEATURE_GLOSSARY[feature]
    if feature.endswith("_isna"):
        base = feature[:-5]
        return f"missing {FEATURE_GLOSSARY.get(base, base)}"
    if feature.endswith("_freq"):
        base = feature[:-5]
        return f"how common {FEATURE_GLOSSARY.get(base, base)} is"
    if feature.startswith("V"):
        return f"anonymised behavioural signal {feature}"
    return feature


def band_risk(prob: float) -> str:
    """Convert a probability into the language a fraud team uses.

    Analysts do not think in probabilities; they think in queues. These bands
    map directly onto the decision tiers used by the Stage 5 risk engine.
    """
    if prob >= 0.90:
        return "CRITICAL"
    if prob >= 0.70:
        return "HIGH"
    if prob >= 0.40:
        return "MEDIUM"
    if prob >= 0.15:
        return "LOW"
    return "MINIMAL"


@dataclass
class CaseNarrative:
    """One investigable case: the alert, its drivers, and its text form."""

    transaction_id: int
    probability: float
    risk_band: str
    amount: float
    is_fraud: int | None
    drivers: list[str] = field(default_factory=list)
    text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "probability": round(self.probability, 4),
            "risk_band": self.risk_band,
            "amount": round(self.amount, 2),
            "is_fraud": self.is_fraud,
            "drivers": self.drivers,
            "text": self.text,
        }


def build_narrative(
    *,
    transaction_id: int,
    probability: float,
    amount: float,
    reason_codes: list,
    context: dict[str, Any] | None = None,
    is_fraud: int | None = None,
) -> CaseNarrative:
    """Compose a case narrative from a score plus its SHAP reason codes.

    Why a deterministic template rather than asking the LLM to write it
    -------------------------------------------------------------------
    Three reasons, all of which matter in a regulated setting:

      * The narrative must be FAITHFUL to the model. A template cannot
        hallucinate a driver that SHAP did not produce.
      * It is reproducible - the same alert always yields the same text, so
        two analysts reading the same case see the same words.
      * It is free and instant, so we can generate hundreds of thousands of
        them to build the search corpus.

    The LLM's job (Stage 9) is to REASON OVER these narratives and the policy
    documents - not to invent the facts. Keeping generation separate from
    reasoning is what keeps the system auditable.
    """
    ctx = context or {}
    band = band_risk(probability)

    up = [r for r in reason_codes if r.shap_value > 0][:4]
    down = [r for r in reason_codes if r.shap_value < 0][:2]

    drivers = [f"{humanise(r.feature)} = {_fmt(r.value)}" for r in up]
    mitigators = [f"{humanise(r.feature)} = {_fmt(r.value)}" for r in down]

    lines = [
        f"ALERT {transaction_id} | risk {band} ({probability:.1%}) | "
        f"amount {amount:,.2f}",
    ]
    if ctx:
        bits = [f"{k}={v}" for k, v in ctx.items() if v is not None and str(v) != "nan"]
        if bits:
            lines.append("Context: " + ", ".join(bits) + ".")

    if drivers:
        lines.append(
            "The model raised this alert primarily because of "
            + _join(drivers) + "."
        )
    if mitigators:
        lines.append("Reducing the score: " + _join(mitigators) + ".")

    lines.append(
        f"Recommended action: {_action(band)}."
    )
    lines.append("[machine-generated from model output; not an analyst write-up]")

    return CaseNarrative(
        transaction_id=transaction_id,
        probability=float(probability),
        risk_band=band,
        amount=float(amount),
        is_fraud=is_fraud,
        drivers=drivers,
        text=" ".join(lines),
    )


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.3g}"
    return str(v)


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


# The authoritative band -> action mapping, mirroring the decision table in
# corpus/policies/01_risk_scoring_and_decisions.md.
#
# This is a CONSTANT rather than something a model reads out of prose. A 3B
# model asked to read that markdown table reported CRITICAL as "hold and
# review within one hour", which is the row for HIGH - it shifted a row. In a
# compliance setting that is the difference between declining a fraudulent
# transaction and letting it stand for an hour.
#
# Rule: never ask a language model to perform a lookup you can perform exactly.
BAND_ACTIONS: dict[str, str] = {
    "CRITICAL": "decline and contact the cardholder immediately",
    "HIGH": "hold for manual review within the hour",
    "MEDIUM": "queue for same-day analyst review",
    "LOW": "monitor; no action unless a further alert fires",
    "MINIMAL": "approve",
}

BAND_OWNER: dict[str, str] = {
    "CRITICAL": "Fraud Operations",
    "HIGH": "Fraud Operations",
    "MEDIUM": "Fraud Operations",
    "LOW": "Automated",
    "MINIMAL": "Automated",
}


def _action(band: str) -> str:
    return BAND_ACTIONS[band]


def build_corpus(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    explainer,
    model,
    feature_cols: list[str],
    *,
    n_cases: int = 2_000,
    id_col: str = "TransactionID",
    amount_col: str = "TransactionAmt",
    target_col: str | None = "isFraud",
    context_cols: tuple[str, ...] = ("ProductCD", "card4", "card6", "DeviceType"),
) -> list[CaseNarrative]:
    """Generate a searchable corpus of case narratives.

    Sampling strategy - deliberately NOT random
    -------------------------------------------
    A random sample of 2,000 transactions would contain ~70 frauds and would
    be dominated by uninteresting approvals. The corpus exists to answer
    "have we seen a case like this before?", so it must be dense in the cases
    an analyst would actually want to find.

    We therefore take the highest-scoring alerts, which is also exactly what a
    real case-management system contains: investigated alerts, not the whole
    transaction stream.
    """
    order = np.argsort(probabilities)[::-1][:n_cases]
    sub = df.iloc[order]
    probs = probabilities[order]

    out: list[CaseNarrative] = []
    for i in range(len(sub)):
        row = sub.iloc[[i]][feature_cols]
        shap_row = explainer.shap_values(row)
        if shap_row.ndim > 1:
            shap_row = shap_row[0]

        from risklens.models.explain import ReasonCode

        top = np.argsort(np.abs(shap_row))[::-1][:6]
        reasons = [
            ReasonCode(
                feature=str(row.columns[j]),
                value=row.iloc[0, j],
                shap_value=float(shap_row[j]),
                direction="increases risk" if shap_row[j] > 0 else "decreases risk",
            )
            for j in top
        ]
        ctx = {c: sub.iloc[i][c] for c in context_cols if c in sub.columns}
        out.append(build_narrative(
            transaction_id=int(sub.iloc[i][id_col]),
            probability=float(probs[i]),
            amount=float(sub.iloc[i][amount_col]),
            reason_codes=reasons,
            context=ctx,
            is_fraud=int(sub.iloc[i][target_col]) if target_col in sub.columns else None,
        ))
        if (i + 1) % 500 == 0:
            log.info("built %d/%d narratives", i + 1, len(sub))

    log.info("corpus complete: %d narratives", len(out))
    return out
