"""Stage 9 - the fraud investigation copilot.

What this is
------------
An agent that investigates an alert the way an analyst would, by calling
tools. Critically, the tools ARE the rest of RiskLens:

    score_transaction     -> Stage 4 XGBoost model
    explain_alert         -> Stage 7 SHAP reason codes
    find_similar_cases    -> Stage 8 semantic search over case narratives
    lookup_policy         -> Stage 8 RAG over the policy corpus
    get_transaction       -> the underlying data

That is the design point worth defending in an interview: the GenAI layer
CONSUMES the ML system rather than replacing it. The model does the
prediction; the LLM does the reading, retrieving and summarising. Neither
does the other's job.

Two execution modes, and why both exist
---------------------------------------
MODE 1 - `investigate()`  DETERMINISTIC WORKFLOW (the default)
    Calls the tools in a fixed, sensible order, then asks the LLM to write a
    summary of what the tools returned.

MODE 2 - `agent_loop()`   TRUE TOOL-CALLING AGENT
    Gives the LLM the tool schemas and lets IT decide what to call, in a loop.

Mode 2 is the more impressive-sounding architecture. Mode 1 is what I default
to, and the reason is honest engineering rather than caution:

  * A 3-billion-parameter local model calls tools unreliably. It invents
    arguments, skips steps, and sometimes answers from memory instead of
    calling anything.
  * In a regulated setting, an investigation that sometimes skips the policy
    check is worse than one that always runs the same five steps.
  * The deterministic workflow is auditable: every investigation touched the
    same evidence, so two cases are comparable.

Being able to explain WHY you chose the less flashy option is worth more than
having used the flashy one.
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from risklens.genai.rag import DEFAULT_MODEL, DEFAULT_NUM_CTX, generate

log = logging.getLogger(__name__)


ANALYST_SYSTEM_PROMPT = """\
You are a fraud investigation assistant supporting a payments risk analyst.

You will be given EVIDENCE gathered by tools: a model score, the features that
drove it, similar historical cases, and the relevant policy extract.

Write a short investigation summary with exactly these sections:

ASSESSMENT: one or two sentences on what this transaction looks like.
KEY DRIVERS: the main reasons the model scored it as it did.
PRECEDENT: what the similar historical cases suggest.
POLICY: what the retrieved policy requires, citing the source file.
RECOMMENDED ACTION: what the analyst should do next.

Hard rules:
- Use ONLY the evidence provided. Do not add facts, numbers or thresholds
  that are not in it.
- If evidence for a section is missing, write "Not available".
- You ADVISE. The analyst decides. Never state that a transaction has been
  declined or blocked.
- Be concise. The analyst is working a queue.\
"""


# =========================================================================
# Tools
# =========================================================================
@dataclass
class ToolResult:
    """One tool call and what it returned - the audit trail unit."""

    name: str
    arguments: dict[str, Any]
    output: Any
    ok: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "error": self.error,
            "output": self.output,
        }


class FraudToolbox:
    """The tools the copilot can call. Each wraps a real RiskLens component.

    Every tool returns plain JSON-serialisable data, because that is what has
    to be pasted into a prompt and what has to be logged for audit.
    """

    def __init__(
        self,
        *,
        model,
        explainer,
        df: pd.DataFrame,
        feature_cols: list[str],
        case_index=None,
        policy_rag=None,
        id_col: str = "TransactionID",
        amount_col: str = "TransactionAmt",
    ):
        self.model = model
        self.explainer = explainer
        self.df = df.set_index(id_col, drop=False)
        self.feature_cols = feature_cols
        self.case_index = case_index
        self.policy_rag = policy_rag
        self.id_col = id_col
        self.amount_col = amount_col

    # ---- tool 1 ---------------------------------------------------------
    def get_transaction(self, transaction_id: int) -> dict[str, Any]:
        """Raw facts about one transaction."""
        if transaction_id not in self.df.index:
            return {"error": f"transaction {transaction_id} not found"}
        row = self.df.loc[[transaction_id]]
        fields = [
            self.amount_col, "ProductCD", "card4", "card6", "DeviceType",
            "P_emaildomain", "hour", "is_night", "n_missing",
        ]
        return {
            "transaction_id": int(transaction_id),
            **{
                f: (None if pd.isna(row.iloc[0][f]) else _plain(row.iloc[0][f]))
                for f in fields if f in row.columns
            },
        }

    # ---- tool 2 ---------------------------------------------------------
    def score_transaction(self, transaction_id: int) -> dict[str, Any]:
        """Run the Stage 4 model."""
        from risklens.genai.narratives import band_risk

        if transaction_id not in self.df.index:
            return {"error": f"transaction {transaction_id} not found"}
        X = self.df.loc[[transaction_id], self.feature_cols]
        prob = float(self.model.predict_proba(X)[:, 1][0])
        return {
            "transaction_id": int(transaction_id),
            "fraud_probability": round(prob, 4),
            "risk_band": band_risk(prob),
        }

    # ---- tool 3 ---------------------------------------------------------
    def explain_alert(self, transaction_id: int, top_k: int = 6) -> dict[str, Any]:
        """Stage 7 SHAP reason codes for one alert."""
        from risklens.genai.narratives import humanise

        if transaction_id not in self.df.index:
            return {"error": f"transaction {transaction_id} not found"}
        X = self.df.loc[[transaction_id], self.feature_cols]
        sv = self.explainer.shap_values(X)
        if sv.ndim > 1:
            sv = sv[0]
        order = np.argsort(np.abs(sv))[::-1][:top_k]
        return {
            "transaction_id": int(transaction_id),
            "drivers": [
                {
                    "factor": humanise(str(X.columns[i])),
                    "raw_feature": str(X.columns[i]),
                    "value": _plain(X.iloc[0, i]),
                    "impact": round(float(sv[i]), 4),
                    "direction": "increases risk" if sv[i] > 0 else "decreases risk",
                }
                for i in order
            ],
        }

    # ---- tool 4 ---------------------------------------------------------
    def find_similar_cases(self, query: str, k: int = 3) -> dict[str, Any]:
        """Stage 8 semantic search over historical case narratives."""
        if self.case_index is None:
            return {"error": "case index not available"}
        hits = self.case_index.search(query, k=k)
        return {
            "query": query,
            "matches": [
                {
                    "transaction_id": h.transaction_id,
                    "similarity": round(h.score, 3),
                    "risk_band": h.risk_band,
                    "confirmed_fraud": h.is_fraud,
                    "summary": h.text[:280],
                }
                for h in hits
            ],
        }

    # ---- tool 5 ---------------------------------------------------------
    def lookup_policy(self, question: str) -> dict[str, Any]:
        """Stage 8 RAG over the policy corpus."""
        if self.policy_rag is None:
            return {"error": "policy RAG not available"}
        ans = self.policy_rag.ask(question)
        return {"question": question, "guidance": ans.answer, "sources": ans.sources}

    # ---- tool 6 ---------------------------------------------------------
    def required_action(self, risk_band: str) -> dict[str, Any]:
        """DETERMINISTIC band -> action lookup. Not RAG, not the LLM.

        Why this exists
        ---------------
        The first copilot run produced a factual error: asked about a CRITICAL
        alert, it reported the action as "hold and review within one hour".
        That is the policy's row for HIGH. CRITICAL requires "decline and
        contact the cardholder immediately". The model had shifted a row while
        reading a markdown table.

        In a compliance setting that is not a cosmetic slip - it is the
        difference between letting a fraudulent transaction stand for an hour
        and stopping it.

        The fix is architectural rather than a better prompt: **never ask a
        language model to perform a lookup you can perform exactly.** The
        band-to-action mapping is a finite table with five rows. Reading it is
        a dictionary access, not a reasoning task.

        So the copilot now receives the authoritative action as a fact, and
        the LLM's job shrinks to explaining and contextualising it. RAG still
        supplies the surrounding narrative policy, where prose comprehension
        genuinely is the right tool.
        """
        from risklens.genai.narratives import BAND_ACTIONS, BAND_OWNER

        band = (risk_band or "").strip().upper()
        if band not in BAND_ACTIONS:
            return {
                "error": f"unknown risk band {risk_band!r}",
                "valid_bands": sorted(BAND_ACTIONS),
            }
        return {
            "risk_band": band,
            "required_action": BAND_ACTIONS[band],
            "owner": BAND_OWNER[band],
            "source": "01_risk_scoring_and_decisions.md (deterministic lookup)",
            "note": "authoritative - read from the table, not inferred by a model",
        }

    # ---- registry -------------------------------------------------------
    def registry(self) -> dict[str, Callable]:
        return {
            "get_transaction": self.get_transaction,
            "score_transaction": self.score_transaction,
            "explain_alert": self.explain_alert,
            "find_similar_cases": self.find_similar_cases,
            "lookup_policy": self.lookup_policy,
            "required_action": self.required_action,
        }

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI-style tool schemas, which Ollama accepts."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_transaction",
                    "description": "Get the raw facts about one transaction by its ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {"transaction_id": {"type": "integer"}},
                        "required": ["transaction_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "score_transaction",
                    "description": "Get the fraud probability and risk band for a transaction.",
                    "parameters": {
                        "type": "object",
                        "properties": {"transaction_id": {"type": "integer"}},
                        "required": ["transaction_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "explain_alert",
                    "description": "Get the features that drove a transaction's fraud score.",
                    "parameters": {
                        "type": "object",
                        "properties": {"transaction_id": {"type": "integer"}},
                        "required": ["transaction_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_similar_cases",
                    "description": "Search historical fraud cases by natural-language description.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_policy",
                    "description": "Look up fraud operations policy guidance.",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                    },
                },
            },
        ]


def _plain(v: Any) -> Any:
    """numpy/pandas scalars -> plain Python, so json.dumps works."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return str(v) if not isinstance(v, (int, float, str, bool)) else v


# =========================================================================
# MODE 1 - deterministic investigation workflow (default)
# =========================================================================
@dataclass
class Investigation:
    transaction_id: int
    evidence: list[ToolResult] = field(default_factory=list)
    summary: str = ""
    model: str = DEFAULT_MODEL

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "tools_called": [e.name for e in self.evidence],
            "evidence": [e.as_dict() for e in self.evidence],
            "summary": self.summary,
            "model": self.model,
        }


def investigate(
    toolbox: FraudToolbox,
    transaction_id: int,
    *,
    model: str = DEFAULT_MODEL,
    write_summary: bool = True,
) -> Investigation:
    """Run a fixed five-step investigation, then have the LLM summarise it.

    The order mirrors the triage sequence in the alert-triage policy: score,
    then drivers, then precedent, then policy. Fixing the order means every
    investigation touches the same evidence and two cases are comparable.
    """
    inv = Investigation(transaction_id=transaction_id, model=model)

    def run(name: str, fn: Callable, **kwargs) -> Any:
        try:
            out = fn(**kwargs)
            ok = not (isinstance(out, dict) and "error" in out)
            inv.evidence.append(ToolResult(name, kwargs, out, ok=ok,
                                           error=out.get("error") if not ok else None))
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning("tool %s failed: %s", name, exc)
            inv.evidence.append(ToolResult(name, kwargs, None, ok=False, error=str(exc)))
            return None

    facts = run("get_transaction", toolbox.get_transaction, transaction_id=transaction_id)
    score = run("score_transaction", toolbox.score_transaction, transaction_id=transaction_id)
    drivers = run("explain_alert", toolbox.explain_alert, transaction_id=transaction_id)

    # Build the similarity query from the model's own top drivers, so the
    # search reflects why THIS alert fired rather than a generic description.
    if drivers and drivers.get("drivers"):
        top = [d["factor"] for d in drivers["drivers"] if d["impact"] > 0][:3]
        query = "fraud alert driven by " + ", ".join(top) if top else "high risk fraud alert"
    else:
        query = "high risk fraud alert"
    similar = run("find_similar_cases", toolbox.find_similar_cases, query=query, k=3)

    band = (score or {}).get("risk_band", "MEDIUM")

    # DETERMINISTIC first: the authoritative action comes from a table lookup,
    # never from the model reading prose. See FraudToolbox.required_action.
    action = run("required_action", toolbox.required_action, risk_band=band)

    # RAG second, for the surrounding narrative policy - prose comprehension
    # is what retrieval is actually good at.
    policy = run(
        "lookup_policy",
        toolbox.lookup_policy,
        question=f"What are the review requirements and evidence standards for "
                 f"a {band} risk transaction?",
    )

    if write_summary:
        inv.summary = _summarise(facts, score, drivers, similar, policy,
                                 action, model=model)
    return inv


def _summarise(facts, score, drivers, similar, policy, action, *, model: str) -> str:
    """Ask the LLM to write the analyst-facing summary from tool output only."""
    evidence = textwrap.dedent(f"""\
        TRANSACTION FACTS
        {json.dumps(facts, indent=2, default=str)}

        MODEL SCORE
        {json.dumps(score, indent=2, default=str)}

        SCORE DRIVERS (from SHAP - faithful to the model)
        {json.dumps(drivers, indent=2, default=str)}

        SIMILAR HISTORICAL CASES
        {json.dumps(similar, indent=2, default=str)}

        REQUIRED ACTION (authoritative table lookup - use this VERBATIM,
        do not infer the action from the policy prose below)
        {json.dumps(action, indent=2, default=str)}

        POLICY GUIDANCE (narrative context only)
        {json.dumps(policy, indent=2, default=str)}
        """)
    prompt = (
        "Write the investigation summary using the sections you were given.\n\n"
        "EVIDENCE\n========\n" + evidence
    )
    try:
        return generate(prompt, system=ANALYST_SYSTEM_PROMPT, model=model, num_predict=500)
    except Exception as exc:  # noqa: BLE001
        log.warning("summary generation failed: %s", exc)
        return f"[summary unavailable: {exc}]"


# =========================================================================
# MODE 2 - true tool-calling agent loop
# =========================================================================
def agent_loop(
    toolbox: FraudToolbox,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    max_turns: int = 6,
) -> dict[str, Any]:
    """Let the model choose which tools to call, in a loop.

    The loop is the standard agentic pattern:

        while the model asks for a tool:
            execute it
            append the result to the conversation
            ask the model again

    We cap `max_turns` because a small model can loop forever calling the same
    tool. That cap is not a detail - it is the difference between an agent and
    a runaway process.
    """
    import ollama

    registry = toolbox.registry()
    messages = [
        {"role": "system", "content":
            "You are a fraud investigation assistant. Use the available tools to "
            "gather evidence before answering. Call one tool at a time. When you "
            "have enough evidence, give a concise answer grounded only in what the "
            "tools returned."},
        {"role": "user", "content": question},
    ]
    trace: list[dict[str, Any]] = []

    for turn in range(max_turns):
        resp = ollama.chat(
            model=model,
            messages=messages,
            tools=toolbox.schemas(),
            options={"temperature": 0.1, "num_ctx": DEFAULT_NUM_CTX},
        )
        msg = resp["message"]
        messages.append(msg)

        calls = msg.get("tool_calls") or []
        if not calls:
            return {
                "question": question,
                "answer": msg.get("content", "").strip(),
                "turns": turn + 1,
                "trace": trace,
                "mode": "agent_loop",
            }

        for call in calls:
            fn = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            log.info("agent -> %s(%s)", fn, args)

            if fn not in registry:
                out = {"error": f"unknown tool {fn}"}
            else:
                try:
                    out = registry[fn](**args)
                except Exception as exc:  # noqa: BLE001
                    out = {"error": str(exc)}

            trace.append({"turn": turn + 1, "tool": fn, "arguments": args, "output": out})
            messages.append({
                "role": "tool",
                "content": json.dumps(out, default=str)[:4000],
            })

    return {
        "question": question,
        "answer": "Investigation did not converge within the turn limit.",
        "turns": max_turns,
        "trace": trace,
        "mode": "agent_loop",
        "note": "hit max_turns - typical of small local models looping on tools",
    }
