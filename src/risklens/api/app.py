"""Stage 10 - the FastAPI scoring and investigation service.

What this stage is really about
-------------------------------
Everything before this ran in a notebook against a static file. Production is
different in one way that breaks most projects: **training/serving skew**.

Training/serving skew is when the features computed at serving time differ
from those computed at training time - a different median for imputation, a
different frequency map, a column in a different order. The model then sees
inputs it was never trained on and degrades silently. No error is raised.

Our defence is structural: the service loads the SAME artefacts the training
run produced.

    models/xgboost.joblib            the fitted model
    models/frequency_encoder.joblib  the fitted encoder, with TRAIN counts
    models/feature_names.joblib      the exact column list, in order

The encoder is not re-fitted. The feature list is not recomputed. If serving
disagrees with training, it is because an artefact is stale - and the /health
endpoint reports which artefacts are loaded so you can tell.

Endpoints
---------
    GET  /health                     what is loaded and ready
    POST /score                      score a raw transaction payload
    GET  /alerts/{id}/explain        SHAP reason codes for a known alert
    POST /search/cases               semantic search over case narratives
    POST /policy/ask                 RAG over the policy corpus
    POST /investigate/{id}           the full copilot investigation
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("risklens.api")

# Loaded once at startup and shared. Loading a model per request would add
# seconds of latency to every call.
STATE: dict[str, Any] = {}


# =========================================================================
# Request / response schemas
# =========================================================================
class TransactionIn(BaseModel):
    """A raw transaction as the payments platform would send it.

    Only TransactionAmt is required. Everything else is optional because real
    payment messages are sparse - and Stage 2 showed that WHICH fields are
    absent is itself signal, so the model is built to handle it.
    """

    TransactionID: int | None = Field(None, description="optional client reference")
    TransactionAmt: float = Field(..., gt=0, description="transaction amount")
    TransactionDT: int | None = Field(None, description="seconds from reference epoch")
    ProductCD: str | None = None
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = Field(None, description="card network")
    card5: float | None = None
    card6: str | None = Field(None, description="debit or credit")
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionAmt": 425.50,
                "TransactionDT": 9_500_000,
                "ProductCD": "C",
                "card4": "visa",
                "card6": "credit",
                "P_emaildomain": "outlook.com",
                "DeviceType": "mobile",
            }
        }
    }


class ScoreOut(BaseModel):
    fraud_probability: float
    risk_band: str
    decision: str
    threshold: float
    model_version: str


class SearchIn(BaseModel):
    query: str = Field(..., min_length=3)
    k: int = Field(3, ge=1, le=20)


class PolicyIn(BaseModel):
    question: str = Field(..., min_length=5)


# =========================================================================
# Lifespan - load artefacts once
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    import joblib

    from risklens.config import load_data_config

    cfg = load_data_config()
    models_dir = cfg.root / "models"
    STATE["cfg"] = cfg

    # --- the model and its EXACT training-time artefacts ---
    for name, path in [
        ("model", models_dir / "xgboost.joblib"),
        ("encoder", models_dir / "frequency_encoder.joblib"),
        ("features", models_dir / "feature_names.joblib"),
    ]:
        if path.is_file():
            STATE[name] = joblib.load(path)
            log.info("loaded %s", path.name)
        else:
            log.warning("MISSING %s - run scripts/run_train.py", path.name)

    # --- operating threshold, chosen by the Stage 5 risk engine ---
    STATE["threshold"] = 0.5
    results = cfg.reports_dir / "stage04_05_model_results.json"
    if results.is_file():
        import json

        payload = json.loads(results.read_text(encoding="utf-8"))
        t = payload.get("xgboost", {}).get("cost_optimal", {}).get("threshold")
        if t:
            STATE["threshold"] = float(t)
            log.info("operating threshold %.4f (cost-optimal)", t)

    # --- SHAP explainer, if the model loaded ---
    if "model" in STATE:
        try:
            import shap

            STATE["explainer"] = shap.TreeExplainer(STATE["model"])
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP unavailable: %s", exc)

    # --- optional GenAI components ---
    from risklens.genai.search import SemanticIndex

    for key, sub in [("case_index", "cases"), ("policy_index", "policy")]:
        d = cfg.root / "indexes" / sub
        if (d / "index.faiss").is_file():
            try:
                STATE[key] = SemanticIndex.load(d)
                log.info("loaded %s index (%d docs)", sub, STATE[key].index.ntotal)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not load %s index: %s", sub, exc)

    if "policy_index" in STATE:
        from risklens.genai.rag import PolicyRAG

        STATE["policy_rag"] = PolicyRAG(STATE["policy_index"])

    yield
    STATE.clear()


app = FastAPI(
    title="RiskLens",
    description=(
        "Fraud risk scoring and investigation API. Built on the public "
        "IEEE-CIS Fraud Detection benchmark - not proprietary data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# =========================================================================
# Feature construction - must mirror training EXACTLY
# =========================================================================
def build_features(payload: TransactionIn) -> pd.DataFrame:
    """Turn one API payload into the exact feature vector the model expects.

    Three steps, in the same order as training:

      1. deterministic row-wise features  (identical code path as training)
      2. the FITTED frequency encoder     (train-time counts, NOT re-fitted)
      3. reindex to the exact training column list, in order

    Step 3 is the one people forget. XGBoost matches features positionally as
    well as by name; a reordered or missing column produces a confident wrong
    answer rather than an error.
    """
    from risklens.features.build import add_deterministic_features

    raw = payload.model_dump()
    raw.setdefault("TransactionID", 0)
    if raw.get("TransactionDT") is None:
        raw["TransactionDT"] = 86_400
    df = pd.DataFrame([raw])

    df = add_deterministic_features(df)

    encoder = STATE.get("encoder")
    if encoder is not None:
        df = encoder.transform(df)

    features: list[str] = STATE["features"]
    # Any column the payload did not supply becomes NaN - which is correct,
    # not a fallback: the model learned how to route missing values, and
    # missingness is genuine signal in this dataset.
    return df.reindex(columns=features)


def band_and_decision(prob: float, threshold: float) -> tuple[str, str]:
    from risklens.genai.narratives import band_risk

    band = band_risk(prob)
    if prob >= max(threshold, 0.90):
        decision = "DECLINE"
    elif prob >= threshold:
        decision = "REVIEW"
    else:
        decision = "APPROVE"
    return band, decision


# =========================================================================
# Endpoints
# =========================================================================
@app.get("/health")
def health() -> dict[str, Any]:
    """What is loaded. Deliberately explicit about what is missing.

    A health endpoint that returns {"status": "ok"} while the model failed to
    load is worse than no health endpoint at all.
    """
    return {
        "status": "ok" if "model" in STATE else "degraded",
        "model_loaded": "model" in STATE,
        "encoder_loaded": "encoder" in STATE,
        "explainer_loaded": "explainer" in STATE,
        "case_index_loaded": "case_index" in STATE,
        "policy_rag_loaded": "policy_rag" in STATE,
        "n_features": len(STATE.get("features", [])),
        "operating_threshold": STATE.get("threshold"),
    }


@app.post("/score", response_model=ScoreOut)
def score(payload: TransactionIn) -> ScoreOut:
    """Score one transaction."""
    if "model" not in STATE:
        raise HTTPException(503, "model not loaded - run scripts/run_train.py")

    X = build_features(payload)
    prob = float(STATE["model"].predict_proba(X)[:, 1][0])
    threshold = STATE["threshold"]
    band, decision = band_and_decision(prob, threshold)

    return ScoreOut(
        fraud_probability=round(prob, 4),
        risk_band=band,
        decision=decision,
        threshold=round(threshold, 4),
        model_version=app.version,
    )


@app.post("/explain")
def explain(payload: TransactionIn) -> dict[str, Any]:
    """Score plus the reason codes that produced it.

    Required by the model-governance policy: every automated decline or hold
    must carry machine-generated reason codes identifying the drivers.
    """
    if "explainer" not in STATE:
        raise HTTPException(503, "explainer not loaded")

    from risklens.genai.narratives import build_narrative, humanise
    from risklens.models.explain import ReasonCode

    X = build_features(payload)
    prob = float(STATE["model"].predict_proba(X)[:, 1][0])
    sv = STATE["explainer"].shap_values(X)
    if sv.ndim > 1:
        sv = sv[0]

    order = np.argsort(np.abs(sv))[::-1][:6]
    reasons = [
        ReasonCode(
            feature=str(X.columns[i]),
            value=X.iloc[0, i],
            shap_value=float(sv[i]),
            direction="increases risk" if sv[i] > 0 else "decreases risk",
        )
        for i in order
    ]
    band, decision = band_and_decision(prob, STATE["threshold"])
    narrative = build_narrative(
        transaction_id=payload.TransactionID or 0,
        probability=prob,
        amount=payload.TransactionAmt,
        reason_codes=reasons,
    )
    return {
        "fraud_probability": round(prob, 4),
        "risk_band": band,
        "decision": decision,
        "reason_codes": [
            {
                "factor": humanise(r.feature),
                "raw_feature": r.feature,
                "impact": round(r.shap_value, 4),
                "direction": r.direction,
            }
            for r in reasons
        ],
        "narrative": narrative.text,
    }


@app.post("/search/cases")
def search_cases(payload: SearchIn) -> dict[str, Any]:
    """Semantic search over historical case narratives."""
    if "case_index" not in STATE:
        raise HTTPException(503, "case index not built - run scripts/run_genai.py")
    hits = STATE["case_index"].search(payload.query, k=payload.k)
    return {"query": payload.query, "results": [h.as_dict() for h in hits]}


@app.post("/policy/ask")
def policy_ask(payload: PolicyIn) -> dict[str, Any]:
    """RAG question answering over the fraud policy corpus."""
    if "policy_rag" not in STATE:
        raise HTTPException(503, "policy RAG not available")
    return STATE["policy_rag"].ask(payload.question).as_dict()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "RiskLens",
        "docs": "/docs",
        "health": "/health",
        "note": "Public IEEE-CIS benchmark data. Not proprietary.",
    }
