"""Interactive demo — Interpretable Multi-Modal Spam Detection Framework.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import re

import numpy as np
import streamlit as st

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.explain import build_shap_explainer, lime_explain, shap_explain_text
from src.features import ENGINEERED_FEATURE_DESCRIPTIONS, engineered_features, ENGINEERED_FEATURE_NAMES
from src.models import SpamDetector

st.set_page_config(
    page_title="Interpretable Spam Detection",
    page_icon="🛡️",
    layout="wide",
)

# ----------------------------------------------------------------- palette
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, BORDER = "#fcfcfb", "rgba(11,11,11,0.10)"
HAM_BLUE, SPAM_RED = "#2a78d6", "#e34948"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
GOOD_TRACK, CRITICAL_TRACK = "#ddf0dd", "#f6dede"

MODALITY_LABEL = {"sms": "SMS", "email": "Email"}
MODEL_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "Hybrid Ensemble"]


# ------------------------------------------------------------- data access
@st.cache_resource(show_spinner="Loading trained models…")
def load_detector(modality: str) -> SpamDetector:
    return SpamDetector.load(MODELS_DIR / f"{modality}_detector.joblib")


@st.cache_resource(show_spinner=False)
def load_shap_explainer(modality: str):
    return build_shap_explainer(load_detector(modality))


@st.cache_data
def load_json(name: str) -> dict:
    path = REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner="Computing LIME explanation (perturbing the message ~1,000 times)…")
def cached_lime(modality: str, text: str) -> list[tuple[str, float]]:
    exp = lime_explain(load_detector(modality), text)
    return [(w, float(v)) for w, v in exp.as_list(label=1)]


def artifacts_ready() -> bool:
    return all((MODELS_DIR / f"{m}_detector.joblib").exists() for m in ("sms", "email"))


def detect_modality(text: str) -> str:
    t = text.strip()
    if len(t) > 280 or "\n" in t or re.search(
            r"\b(subject|dear|regards|sincerely|unsubscribe|attached)\b", t, re.I):
        return "email"
    return "sms"


# ------------------------------------------------------------ HTML helpers
def verdict_card(label: str, spam_prob: float) -> str:
    is_spam = label == "spam"
    icon = "🚨" if is_spam else "✅"
    title = "SPAM / SCAM DETECTED" if is_spam else "LOOKS LEGITIMATE (HAM)"
    wash = "rgba(208,59,59,0.08)" if is_spam else "rgba(12,163,12,0.08)"
    edge = CRITICAL if is_spam else GOOD
    return f"""
    <div style="background:{wash};border:1px solid {BORDER};border-left:4px solid {edge};
                border-radius:10px;padding:18px 22px;margin-bottom:14px;">
      <div style="font-size:15px;font-weight:600;color:{INK};">{icon}&nbsp; {title}</div>
      <div style="font-size:44px;font-weight:600;color:{INK};line-height:1.25;">{spam_prob:.1%}</div>
      <div style="font-size:12.5px;color:{INK2};">ensemble spam probability
        &nbsp;·&nbsp; decision threshold 50%</div>
    </div>"""


def probability_meter(name: str, p: float, weight: float | None) -> str:
    is_spam = p >= 0.5
    fill = CRITICAL if is_spam else GOOD
    track = CRITICAL_TRACK if is_spam else GOOD_TRACK
    icon = "🚩" if is_spam else "✓"
    w = f'<span style="color:{MUTED};">&nbsp;· vote weight {weight:.0%}</span>' if weight else ""
    return f"""
    <div style="margin:7px 0;">
      <div style="display:flex;justify-content:space-between;font-size:12.5px;color:{INK2};">
        <span>{name}{w}</span>
        <span style="color:{INK};font-weight:600;">{icon}&nbsp;{p:.1%} spam</span>
      </div>
      <div style="background:{track};border-radius:99px;height:10px;margin-top:3px;"
           title="{name}: spam probability {p:.1%}">
        <div style="background:{fill};width:{max(p*100,1.2):.1f}%;height:10px;border-radius:99px;"></div>
      </div>
    </div>"""


def highlighted_message(text: str, lime_pairs: list[tuple[str, float]]) -> str:
    """Colour every word by its LIME weight: red pushes spam, blue pushes ham."""
    weights = {w: v for w, v in lime_pairs}
    max_w = max((abs(v) for v in weights.values()), default=1.0) or 1.0
    parts = []
    for token in re.split(r"(\W+)", text[:2500]):
        v = weights.get(token)
        if v is None or token.strip() == "":
            parts.append(html.escape(token))
            continue
        alpha = 0.15 + 0.55 * abs(v) / max_w
        color = f"rgba(227,73,72,{alpha:.2f})" if v > 0 else f"rgba(42,120,214,{alpha:.2f})"
        parts.append(
            f'<mark style="background:{color};padding:1px 3px;border-radius:4px;color:{INK};" '
            f'title="LIME weight {v:+.3f} ({"pushes spam" if v > 0 else "pushes ham"})">'
            f"{html.escape(token)}</mark>")
    body = "".join(parts)
    legend = (f'<span style="color:{MUTED};font-size:11.5px;">'
              f'<mark style="background:rgba(227,73,72,.45);border-radius:3px;">&nbsp;word&nbsp;</mark>'
              f' pushes toward <b>spam</b> &nbsp;·&nbsp; '
              f'<mark style="background:rgba(42,120,214,.45);border-radius:3px;">&nbsp;word&nbsp;</mark>'
              f' pushes toward <b>ham</b></span>')
    return (f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;'
            f'padding:14px 16px;line-height:1.9;font-size:14px;color:{INK};">{body}</div>'
            f'<div style="margin-top:6px;">{legend}</div>')


def lime_bar_chart(pairs: list[tuple[str, float]]):
    import matplotlib.pyplot as plt
    from src.plotting import AXIS, SURFACE as SF, apply_style, style_axes
    apply_style()
    words = [w for w, _ in pairs][::-1]
    vals = np.array([v for _, v in pairs][::-1])
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(words) + 1.2))
    ax.barh(np.arange(len(words)), vals, height=0.62,
            color=[SPAM_RED if v > 0 else HAM_BLUE for v in vals],
            edgecolor=SF, linewidth=2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(np.arange(len(words)), words)
    ax.set_xlabel("LIME weight   ← pushes ham | pushes spam →", fontsize=9)
    style_axes(ax, xgrid=True, ygrid=False)
    fig.tight_layout()
    return fig


def shap_waterfall_fig(modality: str, text: str):
    import matplotlib.pyplot as plt
    import shap
    from src.plotting import apply_style
    apply_style()
    detector = load_detector(modality)
    exp = shap_explain_text(detector, load_shap_explainer(modality), text)
    plt.figure()
    shap.plots.waterfall(exp, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7.4, 5.2)
    fig.set_facecolor("#fcfcfb")
    fig.tight_layout()
    return fig


# ================================================================== layout
st.markdown(
    f"""<div style="margin-bottom:4px;">
    <span style="font-size:30px;font-weight:700;color:{INK};">🛡️ Interpretable Multi-Modal Spam Detection</span><br>
    <span style="color:{INK2};font-size:14.5px;">Hybrid ensemble (Logistic Regression + Random Forest + XGBoost)
    with SHAP &amp; LIME explanations, for SMS and Email.</span></div>""",
    unsafe_allow_html=True)

if not artifacts_ready():
    st.error("Trained models not found. Run **`python -m src.train`** first "
             "(it downloads the datasets, trains both modalities and saves all artefacts).")
    st.stop()

metrics = load_json("metrics.json")
examples = load_json("examples.json")

tab_detect, tab_perf, tab_global, tab_about = st.tabs(
    ["🔍  Detector", "📊  Model performance", "🧠  Global explanations", "ℹ️  About"])

# ------------------------------------------------------------ detector tab
with tab_detect:
    col_in, col_out = st.columns([5, 6], gap="large")

    with col_in:
        st.subheader("Message")
        mode = st.radio("Modality", ["Auto-detect", "SMS", "Email"], horizontal=True,
                        help="Each modality has its own trained ensemble. "
                             "Auto-detect routes by length and structure.")

        ex_options = ["— pick a real example from the held-out test set —"]
        ex_map = {}
        for m in ("sms", "email"):
            for i, ex in enumerate(examples.get(m, []), 1):
                key = f"{MODALITY_LABEL[m]} · {ex['label']} · “{ex['text'][:48]}…”"
                ex_options.append(key)
                ex_map[key] = (m, ex["text"])

        def _apply_example() -> None:
            sel = st.session_state.get("example_sel")
            if sel in ex_map:
                st.session_state["msg_text"] = ex_map[sel][1]

        st.selectbox("Try an example", ex_options, key="example_sel",
                     on_change=_apply_example)
        text = st.text_area("Paste an SMS or e-mail", key="msg_text", height=220,
                            placeholder="e.g.  URGENT! You have won a £1,000 prize. "
                                        "Call 09061701461 now to claim…")
        analyse = st.button("🔎 Analyse message", type="primary", width="stretch")

    if analyse and not text.strip():
        with col_in:
            st.warning("Please enter a message first.")
    elif text.strip():
        modality = detect_modality(text) if mode == "Auto-detect" else mode.lower()
        detector = load_detector(modality)
        result = detector.analyse(text)

        with col_out:
            st.subheader("Prediction")
            st.caption(f"Routed to the **{MODALITY_LABEL[modality]} ensemble**"
                       + (" (auto-detected)" if mode == "Auto-detect" else ""))
            st.markdown(verdict_card(result["label"], result["spam_probability"]),
                        unsafe_allow_html=True)

            meters = [probability_meter(n, result["per_model"][n],
                                        result["ensemble_weights"].get(n))
                      for n in MODEL_ORDER[:3]]
            meters.append(probability_meter("Hybrid Ensemble (weighted vote)",
                                            result["spam_probability"], None))
            st.markdown("".join(meters), unsafe_allow_html=True)
            st.caption("Vote weights are proportional to each model's cross-validated F1 score.")

        st.divider()
        st.subheader("Why? — explanation of this prediction")
        col_lime, col_shap = st.columns(2, gap="large")

        with col_lime:
            st.markdown("##### LIME — explains the **full hybrid ensemble**")
            lime_pairs = cached_lime(modality, text)
            st.markdown(highlighted_message(text, lime_pairs), unsafe_allow_html=True)
            st.pyplot(lime_bar_chart(lime_pairs), width="stretch")
            st.caption("LIME perturbs the message (~1,000 word-dropout variants), watches how the "
                       "ensemble's output changes, and fits a local linear surrogate. "
                       "Weights are per-word contributions to *this* prediction.")

        with col_shap:
            st.markdown("##### SHAP — explains the **XGBoost component** (exact TreeSHAP)")
            with st.spinner("Computing SHAP values…"):
                st.pyplot(shap_waterfall_fig(modality, text), width="stretch")
            st.caption("Additive log-odds contributions: starting from the dataset base rate "
                       "E[f(x)], each feature pushes toward spam (red, right) or ham (blue, left), "
                       "ending at this message's score. UPPER-CASE features are engineered "
                       "statistics; lower-case ones are TF-IDF words (`httpaddr` = a URL, "
                       "`numbr` = a number, `moneysymb` = a money amount).")

        with st.expander("Engineered stylometric features for this message"):
            vals = engineered_features(text)
            st.dataframe(
                {"Feature": ENGINEERED_FEATURE_NAMES,
                 "Value": [round(v, 3) for v in vals],
                 "Meaning": [ENGINEERED_FEATURE_DESCRIPTIONS[n] for n in ENGINEERED_FEATURE_NAMES]},
                width="stretch", hide_index=True)
    else:
        with col_out:
            st.subheader("Prediction")
            st.info("Enter a message (or pick an example) and press **Analyse** to see the "
                    "verdict, every model's vote, and word-level SHAP + LIME explanations.")

# ---------------------------------------------------------- performance tab
with tab_perf:
    for m in ("sms", "email"):
        info = metrics.get("modalities", {}).get(m, {})
        if not info:
            continue
        meta, models = info["meta"], info["models"]
        ens = models["Hybrid Ensemble"]
        st.subheader(f"{MODALITY_LABEL[m]} — {meta['n_messages']:,} messages "
                     f"({meta['spam_share']:.1%} spam)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ensemble accuracy", f"{ens['accuracy']:.2%}")
        c2.metric("Precision (spam)", f"{ens['precision']:.2%}")
        c3.metric("Recall (spam)", f"{ens['recall']:.2%}")
        c4.metric("ROC-AUC", f"{ens['roc_auc']:.4f}")

        rows = []
        for name in MODEL_ORDER:
            mm = models[name]
            rows.append({
                "Model": name, "Accuracy": mm["accuracy"], "Precision": mm["precision"],
                "Recall": mm["recall"], "F1": mm["f1"], "ROC-AUC": mm["roc_auc"],
                "CV F1 (5-fold)": (f"{mm['cv_f1_mean']:.4f} ± {mm['cv_f1_std']:.4f}"
                                   if "cv_f1_mean" in mm else "—"),
            })
        st.dataframe(rows, width="stretch", hide_index=True,
                     column_config={k: st.column_config.NumberColumn(format="%.4f")
                                    for k in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")})

        f1, f2 = st.columns(2)
        f1.image(str(FIGURES_DIR / f"{m}_model_comparison.png"),
                 caption="Test-set metrics (dot = model; the best per panel is labelled)")
        f2.image(str(FIGURES_DIR / f"{m}_roc_curves.png"), caption="ROC curves")
        st.image(str(FIGURES_DIR / f"{m}_confusion_matrices.png"),
                 caption="Confusion matrices on the held-out test set")
        st.divider()

    st.image(str(FIGURES_DIR / "dataset_overview.png"),
             caption="Datasets: UCI SMS Spam Collection and Enron-Spam corpus")

# --------------------------------------------------------------- global tab
with tab_global:
    st.markdown(
        "Global SHAP analysis answers **“what has the model actually learned?”** — computed "
        "with exact TreeSHAP over the XGBoost component on a sample of held-out test messages.")
    for m in ("sms", "email"):
        st.subheader(f"{MODALITY_LABEL[m]} modality")
        g1, g2 = st.columns(2)
        g1.image(str(FIGURES_DIR / f"{m}_shap_beeswarm.png"),
                 caption="Beeswarm: each dot is one test message. Position = impact on the "
                         "spam log-odds; colour = feature value (red high, blue low).")
        g2.image(str(FIGURES_DIR / f"{m}_shap_importance.png"),
                 caption="Global importance: mean |SHAP value| per feature")
        st.image(str(FIGURES_DIR / f"{m}_lime_case_study.png"),
                 caption="LIME case study on a real spam message from the test set")
        st.divider()

# ---------------------------------------------------------------- about tab
with tab_about:
    left, right = st.columns([6, 5], gap="large")
    with left:
        st.subheader("Project")
        st.markdown("""
**An Interpretable Multi-Modal Spam Detection Framework Using Explainable AI and
Machine Learning** — a research project that pairs a *hybrid ensemble* of three
heterogeneous learners with two complementary XAI techniques, so every verdict
ships with a human-readable justification.

- **Modalities** — SMS (UCI SMS Spam Collection, 5.5k msgs) and Email
  (Enron-Spam corpus, 12k msgs). Each modality gets its own feature space and
  its own ensemble; an auto-router picks the right one at inference time.
- **Features** — TF-IDF over normalised text (URLs → `httpaddr`, numbers →
  `numbr`, money → `moneysymb`…) **+ 15 engineered stylometric statistics**
  (SHOUTING ratio, punctuation abuse, URL/phone counts, a transparent
  scam-keyword lexicon…).
- **Models** — Logistic Regression, Random Forest and XGBoost, combined by
  **performance-weighted soft voting** (weights ∝ cross-validated F1).
- **Explainability** — **LIME** explains the full ensemble locally
  (word-level, model-agnostic); **SHAP** (exact TreeSHAP) explains the XGBoost
  component locally (waterfall) and globally (beeswarm / importance).

> ⚠️ Research & education use only — not a production mail filter. False
> positives/negatives are expected on out-of-distribution messages.
""")
        st.subheader("References")
        st.markdown("""
1. Almeida, T. & Gómez Hidalgo, J.M. (2011). *SMS Spam Collection*. UCI ML Repository.
2. Metsis, V., Androutsopoulos, I. & Paliouras, G. (2006). *Spam Filtering with Naive Bayes — Which Naive Bayes?* CEAS.
3. Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
4. Ribeiro, M.T., Singh, S. & Guestrin, C. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier* (LIME). KDD.
5. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
""")
    with right:
        st.subheader("Architecture")
        st.graphviz_chart("""
digraph G {
  rankdir=TB; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#fcfcfb", color="#c3c2b7",
        fontname="Helvetica", fontsize=11, fontcolor="#0b0b0b"];
  edge [color="#898781", arrowsize=0.7];
  in  [label="Raw message (SMS / Email)"];
  rt  [label="Modality router"];
  pre [label="Normalisation\\nURLs→httpaddr · numbers→numbr · money→moneysymb"];
  ft  [label="Feature extraction\\nTF-IDF (1–2 grams) + 15 engineered statistics"];
  lr  [label="Logistic\\nRegression", fillcolor="#e8f1fb"];
  rf  [label="Random\\nForest", fillcolor="#fdeee7"];
  xgb [label="XGBoost", fillcolor="#e4f5ee"];
  ens [label="Hybrid soft-voting ensemble\\n(weights ∝ CV F1)", fillcolor="#fdf3dd"];
  out [label="Verdict + spam probability"];
  xai [label="XAI layer\\nLIME (full ensemble) · SHAP (TreeSHAP)", fillcolor="#f0efec"];
  in -> rt -> pre -> ft;
  ft -> lr; ft -> rf; ft -> xgb;
  lr -> ens; rf -> ens; xgb -> ens;
  ens -> out; ens -> xai; ft -> xai;
}
""")
        info = metrics.get("modalities", {})
        if info:
            st.caption(f"Artefacts generated {metrics.get('generated_at', '')} · "
                       f"SMS features {info['sms']['meta']['n_features']:,} · "
                       f"Email features {info['email']['meta']['n_features']:,}")
