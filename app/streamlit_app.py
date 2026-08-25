"""Stage 10 - the Streamlit analyst console.

Who this is for
---------------
A fraud analyst, not a data scientist. That single constraint drives every
design choice here:

  * Show a RISK BAND and a DECISION, not a raw probability. Analysts route by
    queue, not by float.
  * Show REASON CODES in plain English. "V257 = 3.2" is not actionable;
    "how common this device is" is.
  * Show PRECEDENT. The first question on any alert is "have we seen this
    before?"
  * Show POLICY. The analyst has to justify the decision, and needs the
    citation.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="RiskLens", page_icon="🛡️", layout="wide")

BAND_COLOR = {
    "CRITICAL": "#c0392b",
    "HIGH": "#e67e22",
    "MEDIUM": "#f1c40f",
    "LOW": "#3498db",
    "MINIMAL": "#27ae60",
}


# =========================================================================
# Cached loaders - Streamlit reruns the whole script on every interaction,
# so without caching we would reload a 30 MB model on every click.
# =========================================================================
@st.cache_resource
def load_artifacts():
    import joblib

    from risklens.config import load_data_config

    cfg = load_data_config()
    md = cfg.root / "models"
    out = {"cfg": cfg}
    for k, f in [("model", "xgboost.joblib"),
                 ("encoder", "frequency_encoder.joblib"),
                 ("features", "feature_names.joblib"),
                 # feature_schema carries the TRAINING CategoricalDtypes.
                 # Without it, serving builds its own category list and the
                 # integer codes XGBoost splits on no longer line up - a
                 # silently wrong prediction rather than an error.
                 ("schema", "feature_schema.joblib")]:
        p = md / f
        if p.is_file():
            out[k] = joblib.load(p)
    if "model" in out:
        try:
            import shap

            out["explainer"] = shap.TreeExplainer(out["model"])
        except Exception:  # noqa: BLE001
            pass

    res = cfg.reports_dir / "stage04_05_model_results.json"
    if res.is_file():
        out["results"] = json.loads(res.read_text(encoding="utf-8"))
        out["threshold"] = (
            out["results"].get("xgboost", {}).get("cost_optimal", {}).get("threshold", 0.5)
        )
    else:
        out["threshold"] = 0.5
    return out


@st.cache_resource
def load_indexes():
    from risklens.genai.search import SemanticIndex

    cfg = load_artifacts()["cfg"]
    out = {}
    for key, sub in [("cases", "cases"), ("policy", "policy")]:
        d = cfg.root / "indexes" / sub
        if (d / "index.faiss").is_file():
            try:
                out[key] = SemanticIndex.load(d)
            except Exception:  # noqa: BLE001
                pass
    if "policy" in out:
        from risklens.genai.rag import PolicyRAG

        out["rag"] = PolicyRAG(out["policy"])
    return out


A = load_artifacts()


# =========================================================================
# Sidebar
# =========================================================================
st.sidebar.title("🛡️ RiskLens")
st.sidebar.caption("Fraud risk scoring & investigation")
st.sidebar.markdown("---")

st.sidebar.subheader("System status")
st.sidebar.write("Model", "✅" if "model" in A else "❌")
st.sidebar.write("Schema", "✅" if "schema" in A else "❌")
st.sidebar.write("Explainer", "✅" if "explainer" in A else "❌")
IDX = load_indexes()
st.sidebar.write("Case search", "✅" if "cases" in IDX else "❌")
st.sidebar.write("Policy RAG", "✅" if "rag" in IDX else "❌")
if "features" in A:
    st.sidebar.metric("Features", len(A["features"]))
st.sidebar.metric("Threshold", f"{A['threshold']:.4f}")
st.sidebar.caption("Chosen by minimising expected loss, not by maximising F1.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Public IEEE-CIS benchmark data (Vesta, 2019). **Not proprietary data.**"
)

page = st.sidebar.radio(
    "View", ["Score a transaction", "Model performance", "Case search", "Policy Q&A"]
)


# =========================================================================
# Page 1 - score
# =========================================================================
if page == "Score a transaction":
    st.title("Score a transaction")
    st.caption(
        "Enter transaction details. Fields you leave blank stay missing — "
        "which is realistic, and missingness is itself signal in this model."
    )

    if "model" not in A:
        st.error("No model loaded. Run `python scripts/run_train.py` first.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Amount", min_value=0.01, value=425.50, step=10.0)
        product = st.selectbox("Product code", ["W", "C", "H", "R", "S"], index=1)
        card4 = st.selectbox("Card network", ["visa", "mastercard", "american express",
                                              "discover"], index=0)
    with c2:
        card6 = st.selectbox("Card type", ["credit", "debit"], index=0)
        email = st.selectbox(
            "Payer email domain",
            ["gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "anonymous.com"],
            index=1,
        )
        device = st.selectbox("Device type", ["(missing)", "desktop", "mobile"], index=2)
    with c3:
        hour = st.slider("Hour of day (relative)", 0, 23, 3)
        dist1 = st.number_input("Billing distance", min_value=0.0, value=0.0, step=10.0)
        include_dist = st.checkbox("Include distance", value=False)

    if st.button("Score", type="primary", use_container_width=True):
        from risklens.api.app import STATE, build_features
        from risklens.api.app import TransactionIn

        STATE.update({k: A[k] for k in ("model", "encoder", "features", "schema")
                      if k in A})

        payload = TransactionIn(
            TransactionAmt=amount,
            TransactionDT=hour * 3600 + 86_400,
            ProductCD=product,
            card4=card4,
            card6=card6,
            P_emaildomain=email,
            DeviceType=None if device == "(missing)" else device,
            dist1=dist1 if include_dist else None,
        )
        X = build_features(payload)
        prob = float(A["model"].predict_proba(X)[:, 1][0])

        from risklens.genai.narratives import band_risk

        band = band_risk(prob)
        decision = ("DECLINE" if prob >= max(A["threshold"], 0.90)
                    else "REVIEW" if prob >= A["threshold"] else "APPROVE")

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud probability", f"{prob:.2%}")
        m2.markdown(
            f"<div style='padding:14px;border-radius:8px;text-align:center;"
            f"background:{BAND_COLOR[band]};color:white;font-weight:700;"
            f"font-size:22px'>{band}</div>",
            unsafe_allow_html=True,
        )
        m3.metric("Decision", decision)

        if "explainer" in A:
            import numpy as np

            from risklens.genai.narratives import humanise

            sv = A["explainer"].shap_values(X)
            if sv.ndim > 1:
                sv = sv[0]
            order = np.argsort(np.abs(sv))[::-1][:8]
            rows = [
                {
                    "Factor": humanise(str(X.columns[i])),
                    "Value": X.iloc[0, i],
                    "Impact": float(sv[i]),
                    "Direction": "↑ raises risk" if sv[i] > 0 else "↓ lowers risk",
                }
                for i in order
            ]
            st.subheader("Why — reason codes")
            st.caption(
                "From SHAP, so these are faithful to the model. They are not "
                "an interpretation, and they sum exactly to the prediction."
            )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =========================================================================
# Page 2 - performance
# =========================================================================
elif page == "Model performance":
    st.title("Model performance")
    if "results" not in A:
        st.error("No results found. Run `python scripts/run_train.py` first.")
        st.stop()

    r = A["results"]
    xgb = r.get("xgboost", {})
    base = r.get("baseline_logreg", {})

    st.subheader("Ranking quality")
    c = st.columns(4)
    m = xgb.get("metrics", {})
    c[0].metric("PR-AUC", f"{m.get('pr_auc', 0):.4f}",
                f"{m.get('pr_auc_lift', 0):.1f}× random")
    c[1].metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")
    c[2].metric("Brier", f"{m.get('brier', 0):.5f}", "lower is better")
    c[3].metric("Base rate", f"{m.get('base_rate', 0):.3%}")

    if base:
        st.caption(
            f"Baseline logistic regression PR-AUC: "
            f"{base.get('metrics', {}).get('pr_auc', 0):.4f} — "
            "the number XGBoost had to beat to justify its complexity."
        )

    st.subheader("Operating points")
    t1, t2 = st.tabs(["By precision target", "By alert budget"])
    with t1:
        st.caption("What if the fraud team requires a minimum precision?")
        st.dataframe(pd.DataFrame(xgb.get("at_precision", {})).T,
                     use_container_width=True)
    with t2:
        st.caption("What if the team can only review a fixed share of traffic?")
        st.dataframe(pd.DataFrame(xgb.get("at_alert_budget", {})).T,
                     use_container_width=True)

    st.subheader("Risk engine — threshold chosen by money")
    co = xgb.get("cost_optimal", {})
    dn = xgb.get("do_nothing_baseline", {})
    va = xgb.get("value_added", {})
    c = st.columns(3)
    c[0].metric("Do nothing", f"{dn.get('net_cost', 0):,.0f}", "all fraud gets through")
    c[1].metric("With RiskLens", f"{co.get('net_cost', 0):,.0f}", "net cost")
    c[2].metric("Saving", f"{va.get('net_saving', 0):,.0f}",
                f"{va.get('pct_of_fraud_loss_avoided', 0):.1f}% of loss avoided")
    st.caption(
        "A false negative costs the transaction amount; a false positive costs "
        "about 15 in review time. Asymmetric, and one side varies per "
        "transaction — which is why a single probability threshold is suboptimal."
    )

    figs = A["cfg"].reports_dir / "figures"
    if figs.is_dir():
        st.subheader("Analysis figures")
        for p in sorted(figs.glob("*.png")):
            st.image(str(p), caption=p.stem.replace("_", " "), use_container_width=True)


# =========================================================================
# Page 3 - case search
# =========================================================================
elif page == "Case search":
    st.title("Find similar cases")
    st.caption(
        "Semantic search over historical case narratives. Matches on meaning, "
        "not keywords — 'night-time purchase from a new browser' finds cases "
        "described as 'overnight transaction on an unrecognised device'."
    )
    if "cases" not in IDX:
        st.error("Case index not built. Run `python scripts/run_genai.py` first.")
        st.stop()

    q = st.text_input("Describe the case",
                      "overnight transaction on an unrecognised device with a rare card")
    k = st.slider("Results", 1, 10, 3)
    if st.button("Search", type="primary"):
        for h in IDX["cases"].search(q, k=k):
            fraud = ("🔴 confirmed fraud" if h.is_fraud == 1
                     else "🟢 not fraud" if h.is_fraud == 0 else "⚪ unlabelled")
            with st.expander(
                f"#{h.rank} · similarity {h.score:.3f} · {h.risk_band} · {fraud}"
            ):
                st.write(h.text)


# =========================================================================
# Page 4 - policy RAG
# =========================================================================
elif page == "Policy Q&A":
    st.title("Ask the fraud policy")
    st.caption(
        "Retrieval-augmented generation over the policy corpus. The model may "
        "only answer from retrieved passages and must cite its source."
    )
    if "rag" not in IDX:
        st.error("Policy index not built. Run `python scripts/run_genai.py` first.")
        st.stop()

    st.info(
        "Policy documents are **illustrative**, written for this project. "
        "They are not real policies of any institution."
    )
    q = st.text_input("Question",
                      "What action is required for a HIGH risk transaction?")
    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving and generating..."):
            ans = IDX["rag"].ask(q)
        st.markdown(ans.answer)
        st.caption("Sources: " + ", ".join(ans.sources))
