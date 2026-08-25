"""Stages 6-9 entry point: unsupervised, SHAP, narratives, search, RAG, agent.

Pipeline
--------
    1. load data + the trained model
    2. Stage 7  - SHAP global importance + a leakage audit
    3. Stage 6  - IsolationForest anomaly detection + fraud typology clustering
    4. Stage 8a - generate case narratives from scores + reason codes
    5. Stage 8b - embed narratives and policy docs into FAISS indexes
    6. Stage 8c - RAG question answering over the policy corpus
    7. Stage 9  - a full copilot investigation of one alert

Usage
-----
    python scripts/run_genai.py
    python scripts/run_genai.py --skip-llm     # everything except Ollama calls
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
import pandas as pd  # noqa: E402

from risklens.config import load_data_config  # noqa: E402
from risklens.data.ingest import load_joined  # noqa: E402
from risklens.data.split import temporal_split  # noqa: E402
from risklens.features.build import add_deterministic_features  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage06_09")


def section(t: str) -> None:
    print(f"\n{'=' * 74}\n  {t}\n{'=' * 74}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-llm", action="store_true", help="no Ollama calls")
    ap.add_argument("--n-cases", type=int, default=1500)
    args = ap.parse_args()

    cfg = load_data_config()
    md = cfg.root / "models"
    idx_dir = cfg.root / "indexes"
    findings: dict = {}

    for f in ("xgboost.joblib", "frequency_encoder.joblib", "feature_names.joblib"):
        if not (md / f).is_file():
            log.error("missing %s - run scripts/run_train.py first", f)
            return 1

    model = joblib.load(md / "xgboost.joblib")
    encoder = joblib.load(md / "frequency_encoder.joblib")
    features = joblib.load(md / "feature_names.joblib")

    # ---- rebuild the same feature frame the model was trained on ---------
    log.info("rebuilding features ...")
    df = load_joined(cfg)
    df = add_deterministic_features(df)
    df = encoder.transform(df)
    masks, _, _ = temporal_split(
        df, time_col=cfg.time_column, target_col=cfg.target, split_cfg=cfg.split
    )
    val = df[masks["val"]].reset_index(drop=True)
    X_val = val[features]
    y_val = val[cfg.target].to_numpy()
    log.info("validation partition: %s", X_val.shape)

    probs = model.predict_proba(X_val)[:, 1]

    # =====================================================================
    # STAGE 7 - SHAP
    # =====================================================================
    section("STAGE 7 - EXPLAINABILITY (SHAP)")
    print("SHAP borrows the Shapley value from game theory: if features")
    print("'cooperate' to produce a prediction, how much did each contribute?")
    print("The key property is additivity - base_value + sum(shap) = prediction -")
    print("so an explanation can never omit a contributing factor.\n")

    from risklens.models.explain import (
        compute_shap_values, global_importance, leakage_audit,
    )

    t0 = time.perf_counter()
    explainer, shap_values, X_shap = compute_shap_values(model, X_val, max_rows=3000)
    imp = global_importance(shap_values, X_shap, top=20)
    print(f"computed in {time.perf_counter() - t0:.1f}s\n")
    print(imp.to_string(index=False))

    audit = leakage_audit(imp)
    print(f"\nLEAKAGE AUDIT: {audit['verdict']}")
    print(f"  top feature {audit['top_feature']} holds "
          f"{audit['top_feature_share']:.1%} of total SHAP magnitude")
    imp.to_csv(cfg.reports_dir / "stage07_shap_importance.csv", index=False)
    findings["shap"] = {"top_20": imp.to_dict("records"), "leakage_audit": audit}
    _persist(cfg, findings)

    # =====================================================================
    # STAGE 6 - unsupervised
    # =====================================================================
    section("STAGE 6 - ANOMALY DETECTION (unsupervised, never sees a label)")
    from risklens.models.unsupervised import (
        anomaly_scores, cluster_fraud_typologies, describe_typologies,
        evaluate_anomaly_detector, fit_isolation_forest,
    )

    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(X_val[c])][:80]
    Xa = X_val[num_cols].fillna(-999)
    t0 = time.perf_counter()
    iso = fit_isolation_forest(Xa)
    ascore = anomaly_scores(iso, Xa)
    iso_eval = evaluate_anomaly_detector(ascore, y_val, float(y_val.mean()))
    print(f"fitted in {time.perf_counter() - t0:.1f}s")
    print(json.dumps(iso_eval, indent=2))
    print("\nThis is EXPECTED to be far worse than the supervised model - it")
    print("had zero labels. The question is whether it beats random, because")
    print("that means fraud is genuinely anomalous in feature space and the")
    print("detector is worth keeping as a safety net for novel attack types.")
    findings["isolation_forest"] = iso_eval
    _persist(cfg, findings)

    section("STAGE 6 - FRAUD TYPOLOGIES (clustering confirmed fraud)")
    fraud_rows = val[val[cfg.target] == 1].reset_index(drop=True)
    print(f"clustering {len(fraud_rows):,} confirmed fraud cases")
    print("Note we cluster ONLY fraud. Clustering everything would just")
    print("rediscover the majority class; we are asking a different question:")
    print("what KINDS of fraud are there?\n")

    tcols = [c for c in num_cols if c in fraud_rows.columns][:40]
    km, pipe, labels = cluster_fraud_typologies(fraud_rows, tcols, n_clusters=5)
    typologies = describe_typologies(fraud_rows, labels, tcols,
                                     amount_col=cfg.amount_column)
    for t in typologies:
        print(f"  Typology {t.cluster_id}: {t.n_cases:>5,} cases ({t.share:.1%})  "
              f"avg amount {t.avg_amount:>9,.2f}")
        print(f"     signature: {t.label}")
    findings["typologies"] = [t.as_dict() for t in typologies]
    _persist(cfg, findings)

    # =====================================================================
    # STAGE 8a - narratives
    # =====================================================================
    section("STAGE 8a - CASE NARRATIVES (structured -> text)")
    print("The dataset has no free text, so rather than inventing a fake text")
    print("column we GENERATE the artefact a real fraud operation produces:")
    print("the case narrative an analyst writes when investigating an alert.")
    print("That gives a genuine, non-contrived NLP corpus.\n")

    from risklens.genai.narratives import build_corpus

    t0 = time.perf_counter()
    corpus = build_corpus(
        val, probs, explainer, model, features,
        n_cases=args.n_cases, target_col=cfg.target, amount_col=cfg.amount_column,
    )
    print(f"built {len(corpus):,} narratives in {time.perf_counter() - t0:.1f}s")
    print(f"\nEXAMPLE:\n{corpus[0].text}\n")
    n_fraud = sum(1 for c in corpus if c.is_fraud == 1)
    print(f"corpus contains {n_fraud:,} confirmed fraud of {len(corpus):,} "
          f"({n_fraud / len(corpus):.1%}) - far denser than the 3.5% base rate,")
    print("because we took the highest-scoring alerts. That is deliberate: a")
    print("real case-management system holds investigated alerts, not all traffic.")

    with open(cfg.reports_dir / "stage08_narratives_sample.json", "w",
              encoding="utf-8") as fh:
        json.dump([c.as_dict() for c in corpus[:50]], fh, indent=2)

    # =====================================================================
    # STAGE 8b - indexes
    # =====================================================================
    section("STAGE 8b - SEMANTIC SEARCH (embeddings + FAISS)")
    from risklens.genai.search import SemanticIndex, load_policy_corpus

    t0 = time.perf_counter()
    case_index = SemanticIndex().build(
        [c.text for c in corpus], [c.as_dict() for c in corpus]
    )
    case_index.save(idx_dir / "cases")
    print(f"case index: {case_index.index.ntotal:,} vectors in "
          f"{time.perf_counter() - t0:.1f}s")

    texts, meta = load_policy_corpus(cfg.root / "corpus" / "policies")
    policy_index = SemanticIndex().build(texts, meta)
    policy_index.save(idx_dir / "policy")
    print(f"policy index: {policy_index.index.ntotal} chunks")

    print("\n--- semantic search demo ---")
    q = "overnight transaction on an unrecognised device with a rare card"
    print(f'query: "{q}"')
    for h in case_index.search(q, k=3):
        print(f"  #{h.rank} sim={h.score:.3f} band={h.risk_band} "
              f"fraud={h.is_fraud} | {h.text[:110]}...")
    print("\nNote: no keyword from the query need appear in the results.")
    print("Matching is on MEANING, via 384-dimensional sentence embeddings.")
    findings["search_demo"] = {"query": q,
                               "hits": [h.as_dict() for h in case_index.search(q, k=3)]}
    _persist(cfg, findings)

    if args.skip_llm:
        _persist(cfg, findings)
        section("DONE (LLM stages skipped)")
        return 0

    # =====================================================================
    # STAGE 8c - RAG
    # =====================================================================
    section("STAGE 8c - RAG OVER FRAUD POLICY")
    from risklens.genai.rag import PolicyRAG, groundedness_check, ollama_available

    if not ollama_available():
        print("Ollama not available - skipping RAG and agent stages.")
        _persist(cfg, findings)
        return 0

    rag = PolicyRAG(policy_index)
    questions = [
        "What action is required for a HIGH risk band transaction and who owns it?",
        "How should thresholds be set, and why is F1 not appropriate?",
        "What is the label maturity window and why does it matter for training?",
        "What is the capital of France?",   # deliberately out of scope
    ]
    rag_results = []
    for q in questions:
        print(f"\nQ: {q}")
        t0 = time.perf_counter()
        ans = rag.ask(q)
        hits = rag.retrieve(q)
        g = groundedness_check(ans.answer, [h["text"] for h in hits])
        print(f"A: {ans.answer}")
        print(f"   sources: {', '.join(ans.sources)}")
        print(f"   groundedness: {g['grounded_ratio']:.2f} - {g['verdict']}")
        print(f"   ({time.perf_counter() - t0:.1f}s)")
        rag_results.append({**ans.as_dict(), "groundedness": g})
    findings["rag"] = rag_results
    print("\nThe last question is out of scope on purpose. A correctly")
    print("configured RAG system should refuse it rather than answer from")
    print("general knowledge - that refusal is the safety property we want.")

    # =====================================================================
    # STAGE 9 - the copilot
    # =====================================================================
    section("STAGE 9 - INVESTIGATION COPILOT")
    from risklens.genai.agent import FraudToolbox, investigate

    toolbox = FraudToolbox(
        model=model, explainer=explainer, df=val, feature_cols=features,
        case_index=case_index, policy_rag=rag,
    )
    target_id = int(val.iloc[int(np.argmax(probs))][cfg.join_key])
    print(f"investigating the highest-scoring alert: {target_id}")
    print("Tools available: get_transaction, score_transaction, explain_alert,")
    print("find_similar_cases, lookup_policy — each wraps a real RiskLens")
    print("component, so the GenAI layer CONSUMES the ML system.\n")

    t0 = time.perf_counter()
    inv = investigate(toolbox, target_id)
    print(f"tools called: {' -> '.join(e.name for e in inv.evidence)}")
    print(f"({time.perf_counter() - t0:.1f}s)\n")
    print(inv.summary)
    findings["investigation"] = inv.as_dict()

    _persist(cfg, findings)
    section("DONE")
    print("  indexes -> indexes/")
    print("  results -> reports/stage06_09_genai_results.json")
    return 0


def _persist(cfg, findings: dict) -> None:
    """Write findings, MERGING with whatever is already on disk.

    Two lessons are baked into this function, both learned the hard way.

    1. MERGE, do not overwrite. `run_llm.py` writes the RAG and copilot
       results into the same artefact. A plain overwrite here would silently
       delete them.

    2. Call this after EVERY stage, not once at the end. The first run of
       this script computed SHAP, the anomaly detector and the typologies,
       then crashed in the RAG stage on an Ollama out-of-memory error - and
       because persistence happened only at the end, every earlier result was
       lost. Minutes of compute discarded by a failure in an unrelated stage.
    """
    path = cfg.reports_dir / "stage06_09_genai_results.json"
    existing: dict = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("existing results file unreadable; starting fresh")
    existing.update(findings)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    log.info("persisted %d result sections", len(existing))


if __name__ == "__main__":
    raise SystemExit(main())
