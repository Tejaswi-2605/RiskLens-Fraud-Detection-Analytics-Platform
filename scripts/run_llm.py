"""Stages 8c-9 - RAG and the copilot, reusing already-built indexes.

Why this is separate from run_genai.py
--------------------------------------
Building the case corpus requires 1,500 individual SHAP explanations and takes
minutes. The indexes are then saved to disk. If only the LLM stages fail - as
they did, on an Ollama out-of-memory error - re-running the whole pipeline to
retry two stages is wasteful.

This script loads the saved FAISS indexes and runs only the parts that call
the model.

Usage
-----
    python scripts/run_llm.py
    python scripts/run_llm.py --agent-loop    # also try the tool-calling loop
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from risklens.config import load_data_config  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("stage08c_09")


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n  {t}\n{'=' * 74}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-loop", action="store_true",
                    help="also demonstrate the true tool-calling loop")
    args = ap.parse_args()

    cfg = load_data_config()
    idx_dir = cfg.root / "indexes"

    from risklens.genai.rag import (
        DEFAULT_MODEL, DEFAULT_NUM_CTX, PolicyRAG, groundedness_check,
        ollama_available,
    )
    from risklens.genai.search import SemanticIndex

    if not ollama_available():
        log.error("Ollama not reachable. Start it, then re-run.")
        return 1
    if not (idx_dir / "policy" / "index.faiss").is_file():
        log.error("policy index missing - run scripts/run_genai.py first")
        return 1

    log.info("model %s, context window %d tokens", DEFAULT_MODEL, DEFAULT_NUM_CTX)
    print(f"\nNOTE: num_ctx is pinned to {DEFAULT_NUM_CTX}. Without it Ollama")
    print("sizes the KV cache to llama3.2's full 128k context, which needs")
    print("~12.3 GB and fails on a 16 GB machine with an opaque HTTP 500.\n")

    policy_index = SemanticIndex.load(idx_dir / "policy")
    case_index = SemanticIndex.load(idx_dir / "cases")
    log.info("loaded %d policy chunks, %d case narratives",
             policy_index.index.ntotal, case_index.index.ntotal)

    findings: dict = {}
    rag = PolicyRAG(policy_index)

    # =====================================================================
    # STAGE 8c - RAG
    # =====================================================================
    section("STAGE 8c - RAG OVER FRAUD POLICY")
    print("Retrieve relevant policy passages, augment the prompt with them,")
    print("generate an answer grounded in them. The model stops being a")
    print("knowledge source and becomes a reading-comprehension engine over")
    print("documents we control.\n")

    questions = [
        "What action is required for a HIGH risk band transaction and who owns it?",
        "How should decision thresholds be set, and why is maximising F1 not appropriate?",
        "What is the label maturity window and why does it matter for training data?",
        "What are the primary indicators of account takeover?",
        "What is the capital of France?",   # deliberately out of scope
    ]

    rag_results = []
    for q in questions:
        print(f"\n{'-' * 70}\nQ: {q}")
        t0 = time.perf_counter()
        try:
            hits = rag.retrieve(q)
            ans = rag.ask(q)
        except Exception as exc:  # noqa: BLE001
            print(f"   FAILED: {exc}")
            rag_results.append({"question": q, "error": str(exc)})
            continue
        g = groundedness_check(ans.answer, [h["text"] for h in hits])
        print(f"A: {ans.answer}")
        print(f"\n   sources      : {', '.join(ans.sources)}")
        print(f"   groundedness : {g['grounded_ratio']:.2f}  ({g['verdict']})")
        print(f"   latency      : {time.perf_counter() - t0:.1f}s")
        rag_results.append({**ans.as_dict(), "groundedness": g,
                            "latency_s": round(time.perf_counter() - t0, 1)})

    findings["rag"] = rag_results
    print(f"\n{'-' * 70}")
    print("The last question is out of scope ON PURPOSE. A correctly")
    print("configured RAG system should REFUSE it rather than answer from")
    print("general knowledge. That refusal is the safety property we want:")
    print("a system that answers off-corpus questions will also invent policy.")

    # =====================================================================
    # STAGE 9 - the copilot
    # =====================================================================
    section("STAGE 9 - INVESTIGATION COPILOT")

    for f in ("xgboost.joblib", "frequency_encoder.joblib", "feature_names.joblib"):
        if not (cfg.root / "models" / f).is_file():
            log.error("missing models/%s", f)
            return 1

    import pandas as pd  # noqa: F401

    from risklens.data.ingest import load_joined
    from risklens.data.split import temporal_split
    from risklens.features.build import add_deterministic_features
    from risklens.features.entity import build_entity_features

    log.info("rebuilding the validation frame ...")
    model = joblib.load(cfg.root / "models" / "xgboost.joblib")
    encoder = joblib.load(cfg.root / "models" / "frequency_encoder.joblib")
    features = joblib.load(cfg.root / "models" / "feature_names.joblib")

    df = load_joined(cfg)
    df = add_deterministic_features(df)
    df = build_entity_features(df, time_col=cfg.time_column,
                               amount_col=cfg.amount_column)
    df = encoder.transform(df)
    masks, _, _ = temporal_split(df, time_col=cfg.time_column,
                                 target_col=cfg.target, split_cfg=cfg.split)
    val = df[masks["val"]].reset_index(drop=True)
    del df

    import shap

    explainer = shap.TreeExplainer(model)
    probs = model.predict_proba(val[features])[:, 1]

    from risklens.genai.agent import FraudToolbox, investigate

    toolbox = FraudToolbox(
        model=model, explainer=explainer, df=val, feature_cols=features,
        case_index=case_index, policy_rag=rag,
    )

    print("Tools available - each wraps a real RiskLens component:")
    for name in toolbox.registry():
        print(f"  {name}")
    print("\nThe GenAI layer CONSUMES the ML system rather than replacing it.\n")

    target_id = int(val.iloc[int(np.argmax(probs))][cfg.join_key])
    print(f"Investigating the highest-scoring alert: {target_id} "
          f"(probability {probs.max():.4f})\n")

    t0 = time.perf_counter()
    inv = investigate(toolbox, target_id)
    print(f"tools called: {' -> '.join(e.name for e in inv.evidence)}")
    print(f"elapsed: {time.perf_counter() - t0:.1f}s\n")
    print("-" * 70)
    print(inv.summary)
    print("-" * 70)
    findings["investigation"] = inv.as_dict()

    if args.agent_loop:
        section("STAGE 9b - TRUE TOOL-CALLING AGENT LOOP (comparison)")
        print("The LLM decides which tools to call. Kept as a comparison,")
        print("not the default: a 3B model calls tools unreliably, and an")
        print("investigation that sometimes skips the policy check is worse")
        print("than one that always runs the same five steps.\n")
        from risklens.genai.agent import agent_loop

        t0 = time.perf_counter()
        out = agent_loop(toolbox, f"Investigate transaction {target_id} and "
                                  "tell me whether it looks like fraud.")
        print(f"turns: {out['turns']}, elapsed {time.perf_counter() - t0:.1f}s")
        print(f"tools the model chose: "
              f"{[t['tool'] for t in out['trace']] or 'NONE - it answered without calling anything'}")
        print(f"\n{out['answer']}")
        findings["agent_loop"] = out

    # ---- persist, merging into the existing genai artefact --------------
    path = cfg.reports_dir / "stage06_09_genai_results.json"
    existing = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    existing.update(findings)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")

    section("DONE")
    print(f"  results -> reports/{path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
