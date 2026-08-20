"""End-to-end training, evaluation and reporting pipeline.

Run with:  ``python -m src.train``

For each modality (SMS, Email) it
  1. splits the data (stratified 80/20),
  2. fits the feature pipeline on the training fold only,
  3. cross-validates the three base learners (5-fold F1) — the CV scores
     become the soft-voting weights of the hybrid ensemble,
  4. fits Logistic Regression, Random Forest and XGBoost,
  5. evaluates all four systems on the held-out test set,
  6. produces publication-style figures (model comparison, ROC, confusion
     matrices, global SHAP summary, a LIME case study),
  7. saves a ready-to-use ``SpamDetector`` bundle for the demo app.

Outputs: ``models/*.joblib``, ``reports/metrics.json``, ``reports/results.md``,
``reports/examples.json`` and ``reports/figures/*.png``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .config import (CV_FOLDS, FIGURES_DIR, MODALITIES, MODEL_DISPLAY_ORDER,
                     MODELS_DIR, RANDOM_STATE, REPORTS_DIR,
                     SHAP_BACKGROUND_SIZE, TEST_SIZE)
from .data_loader import load_processed
from .explain import build_shap_explainer, lime_explain, shap_explain_matrix
from .features import SpamFeaturizer
from .models import HybridVotingEnsemble, SpamDetector, build_base_models
from .plotting import (AXIS, CLASS_COLORS, INK, INK_2, MODEL_COLORS, MUTED,
                       SEQ_BLUE, SURFACE, apply_style, save_fig, style_axes)

MODALITY_LABEL = {"sms": "SMS", "email": "Email"}
METRIC_FNS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}


# ------------------------------------------------------------------ figures
def fig_dataset_overview(dfs: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # (a) class balance per modality — identity via legend + cap labels
    ax = axes[0]
    mods = list(dfs)
    x = np.arange(len(mods))
    width = 0.34
    for i, cls in enumerate(("ham", "spam")):
        counts = [int((dfs[m]["label"] == cls).sum()) for m in mods]
        bars = ax.bar(x + (i - 0.5) * width, counts, width * 0.94,
                      color=CLASS_COLORS[cls], label=cls,
                      edgecolor=SURFACE, linewidth=2)
        for b, c in zip(bars, counts):
            ax.annotate(f"{c:,}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8.5, color=INK_2,
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, [MODALITY_LABEL[m] for m in mods])
    ax.set_title("Class balance by modality")
    ax.set_ylabel("Messages")
    ax.legend(title=None)
    style_axes(ax)
    ax.set_ylim(0, max((dfs[m]["label"] == c).sum() for m in mods for c in ("ham", "spam")) * 1.18)

    # (b) message length by modality and class (log scale)
    ax = axes[1]
    data, positions, colors = [], [], []
    pos = 0
    ticks, tick_labels = [], []
    for m in mods:
        for cls in ("ham", "spam"):
            wc = dfs[m].loc[dfs[m]["label"] == cls, "text"].str.split().str.len()
            data.append(np.log10(np.clip(wc, 1, None)))
            positions.append(pos)
            colors.append(CLASS_COLORS[cls])
            pos += 1
        ticks.append(np.mean(positions[-2:]))
        tick_labels.append(MODALITY_LABEL[m])
        pos += 0.6
    bp = ax.boxplot(data, positions=positions, widths=0.62, patch_artist=True,
                    showfliers=False, medianprops={"color": INK, "linewidth": 1.4},
                    whiskerprops={"color": AXIS}, capprops={"color": AXIS})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(2)
    ax.set_xticks(ticks, tick_labels)
    ax.set_yticks([0, 1, 2, 3], ["1", "10", "100", "1,000"])
    ax.set_ylabel("Words per message")
    ax.set_title("Message length by modality and class")
    style_axes(ax)
    fig.suptitle("")
    fig.tight_layout()
    save_fig(fig, FIGURES_DIR / "dataset_overview.png")


def fig_model_dotplot(modality: str, metrics: dict) -> None:
    """Small-multiple Cleveland dot plot — honest with near-1.0 scores."""
    names = list(METRIC_FNS)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.0), sharex=True)
    all_vals = [metrics[m][k] for m in MODEL_DISPLAY_ORDER for k in names]
    lo = min(all_vals)
    xlim = (max(0.0, lo - 0.012), min(1.004, max(all_vals) + 0.012))
    for ax, metric in zip(axes.ravel(), names):
        vals = [metrics[m][metric] for m in MODEL_DISPLAY_ORDER]
        y = np.arange(len(MODEL_DISPLAY_ORDER))[::-1]
        ax.hlines(y, xlim[0], vals, color=AXIS, linewidth=1.0, zorder=1)
        best = int(np.argmax(vals))
        for i, (model, v) in enumerate(zip(MODEL_DISPLAY_ORDER, vals)):
            ax.scatter(v, y[i], s=110, color=MODEL_COLORS[model], zorder=3,
                       edgecolor=SURFACE, linewidth=2)
            if i == best:  # direct-label the extreme only; table carries the rest
                ax.annotate(f"{v:.3f}", (v, y[i]), xytext=(6, 8),
                            textcoords="offset points", fontsize=8.5,
                            color=INK_2, fontweight="bold")
        ax.set_yticks(y, MODEL_DISPLAY_ORDER)
        ax.set_title(metric.capitalize(), loc="left")
        ax.set_xlim(*xlim)
        style_axes(ax, xgrid=True, ygrid=False)
    fig.suptitle(f"{MODALITY_LABEL[modality]} — test-set performance", x=0.02,
                 ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, FIGURES_DIR / f"{modality}_model_comparison.png")


def fig_roc(modality: str, roc_data: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1.0, zorder=1)
    for model in MODEL_DISPLAY_ORDER:
        fpr, tpr, roc_auc = roc_data[model]
        ax.plot(fpr, tpr, color=MODEL_COLORS[model], linewidth=2,
                label=f"{model}  (AUC {roc_auc:.4f})")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"{MODALITY_LABEL[modality]} — ROC curves")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.02)
    style_axes(ax, xgrid=False, ygrid=True)
    save_fig(fig, FIGURES_DIR / f"{modality}_roc_curves.png")


def fig_confusion(modality: str, metrics: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.6))
    vmax = max(max(map(max, metrics[m]["confusion"])) for m in MODEL_DISPLAY_ORDER)
    for ax, model in zip(axes.ravel(), MODEL_DISPLAY_ORDER):
        cm = np.array(metrics[model]["confusion"])
        ax.imshow(cm, cmap=SEQ_BLUE, vmin=0, vmax=vmax)
        total = cm.sum()
        for r in range(2):
            for c in range(2):
                frac = cm[r, c] / vmax
                color = "#ffffff" if frac > 0.55 else INK
                ax.text(c, r, f"{cm[r, c]:,}\n{cm[r, c] / total:.1%}",
                        ha="center", va="center", fontsize=10, color=color)
        # 2px surface gap between cells
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=3)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks([0, 1], ["ham", "spam"])
        ax.set_yticks([0, 1], ["ham", "spam"])
        ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
        ax.set_ylabel("Actual", color=MUTED, fontsize=9)
        ax.set_title(model, fontsize=10)
        ax.grid(which="major", visible=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"{MODALITY_LABEL[modality]} — confusion matrices (test set)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, FIGURES_DIR / f"{modality}_confusion_matrices.png")


def fig_shap_global(modality: str, detector: SpamDetector, texts: list[str]) -> None:
    explainer = build_shap_explainer(detector)
    exp = shap_explain_matrix(detector, explainer, texts)

    # beeswarm (SHAP's own red/blue diverging convention)
    plt.figure()
    shap.plots.beeswarm(exp, max_display=14, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.4, 5.6)
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.xaxis.label.set_color(INK_2)
    fig.suptitle(f"{MODALITY_LABEL[modality]} — SHAP beeswarm "
                 f"(XGBoost component, {len(texts)} test messages)",
                 x=0.02, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, FIGURES_DIR / f"{modality}_shap_beeswarm.png")

    # global importance — mean |SHAP|, one series → single hue
    imp = np.abs(exp.values).mean(axis=0)
    order = np.argsort(imp)[-15:]
    names = [detector.feature_names[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.barh(np.arange(len(order)), imp[order], height=0.66, color="#2a78d6",
            edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(np.arange(len(order)), names)
    ax.set_xlabel("mean |SHAP value|  (impact on spam log-odds)")
    ax.set_title(f"{MODALITY_LABEL[modality]} — global feature importance (SHAP)")
    style_axes(ax, xgrid=True, ygrid=False)
    save_fig(fig, FIGURES_DIR / f"{modality}_shap_importance.png")


def fig_lime_case(modality: str, detector: SpamDetector, text: str) -> None:
    exp = lime_explain(detector, text)
    pairs = exp.as_list(label=1)
    words = [w for w, _ in pairs][::-1]
    weights = np.array([w for _, w in pairs][::-1])
    colors = [CLASS_COLORS["spam"] if w > 0 else CLASS_COLORS["ham"] for w in weights]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(np.arange(len(words)), weights, height=0.62, color=colors,
            edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(np.arange(len(words)), words)
    spam_p = detector.predict_proba_texts([text])[0, 1]
    excerpt = (text[:70] + "…") if len(text) > 70 else text
    ax.set_title(f'{MODALITY_LABEL[modality]} — LIME case study  '
                 f'(ensemble spam probability {spam_p:.1%})\n"{excerpt}"',
                 fontsize=10)
    ax.set_xlabel("LIME weight   ← pushes ham   |   pushes spam →")
    style_axes(ax, xgrid=True, ygrid=False)
    save_fig(fig, FIGURES_DIR / f"{modality}_lime_case_study.png")


# ----------------------------------------------------------------- training
def train_modality(modality: str) -> dict:
    t0 = time.time()
    label = MODALITY_LABEL[modality]
    print(f"\n=== {label} modality " + "=" * 46)
    df = load_processed(modality)
    texts = df["text"].tolist()
    y = (df["label"] == "spam").to_numpy(dtype=int)

    idx = np.arange(len(texts))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    texts_train = [texts[i] for i in idx_train]
    texts_test = [texts[i] for i in idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    print(f"  featurising {len(texts_train):,} train / {len(texts_test):,} test messages…")
    featurizer = SpamFeaturizer(modality=modality).fit(texts_train)
    X_train = featurizer.transform(texts_train)
    X_test = featurizer.transform(texts_test)
    print(f"  feature space: {X_train.shape[1]:,} features "
          f"({featurizer.n_tfidf_features_:,} TF-IDF + "
          f"{X_train.shape[1] - featurizer.n_tfidf_features_} engineered)")

    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    base_models = build_base_models(scale_pos_weight=spw)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores, weights = {}, {}
    for name, model in base_models.items():
        t = time.time()
        scores = cross_val_score(model, X_train, y_train, scoring="f1", cv=skf)
        cv_scores[name] = (float(scores.mean()), float(scores.std()))
        weights[name] = float(scores.mean())
        print(f"  CV F1 {name:<20} {scores.mean():.4f} ± {scores.std():.4f}"
              f"  ({time.time() - t:.1f}s)")

    for name, model in base_models.items():
        t = time.time()
        model.fit(X_train, y_train)
        print(f"  fitted {name:<20} ({time.time() - t:.1f}s)")

    ensemble = HybridVotingEnsemble(base_models, weights)

    # ------------------------------------------------------------ evaluate
    metrics, roc_data = {}, {}
    systems = dict(base_models)
    systems["Hybrid Ensemble"] = ensemble
    for name, system in systems.items():
        proba = system.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        entry = {k: float(fn(y_test, pred)) for k, fn in METRIC_FNS.items()}
        fpr, tpr, _ = roc_curve(y_test, proba)
        entry["roc_auc"] = float(auc(fpr, tpr))
        entry["confusion"] = confusion_matrix(y_test, pred).tolist()
        if name in cv_scores:
            entry["cv_f1_mean"], entry["cv_f1_std"] = cv_scores[name]
        metrics[name] = entry
        roc_data[name] = (fpr, tpr, entry["roc_auc"])
        print(f"  test  {name:<20} acc {entry['accuracy']:.4f}  "
              f"P {entry['precision']:.4f}  R {entry['recall']:.4f}  "
              f"F1 {entry['f1']:.4f}  AUC {entry['roc_auc']:.4f}")

    detector = SpamDetector(
        modality=modality, featurizer=featurizer, base_models=base_models,
        ensemble=ensemble,
        metadata={
            "n_messages": int(len(df)),
            "n_train": int(len(idx_train)),
            "n_test": int(len(idx_test)),
            "spam_share": float(y.mean()),
            "n_features": int(X_train.shape[1]),
            "cv_f1": {k: v[0] for k, v in cv_scores.items()},
            "test_metrics": {m: {k: metrics[m][k] for k in
                                 ("accuracy", "precision", "recall", "f1", "roc_auc")}
                             for m in metrics},
        })
    out = MODELS_DIR / f"{modality}_detector.joblib"
    detector.save(out)
    print(f"  saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # ------------------------------------------------------------- figures
    print("  rendering figures…")
    fig_model_dotplot(modality, metrics)
    fig_roc(modality, roc_data)
    fig_confusion(modality, metrics)

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(texts_test), size=min(SHAP_BACKGROUND_SIZE, len(texts_test)),
                            replace=False)
    fig_shap_global(modality, detector, [texts_test[i] for i in sample_idx])

    spam_test = [t for t, yy in zip(texts_test, y_test) if yy == 1]
    case = max(spam_test[:200], key=lambda t: detector.predict_proba_texts([t])[0, 1])
    fig_lime_case(modality, detector, case)

    # example messages for the demo app (held-out test set only)
    examples = []
    for cls, cls_id in (("spam", 1), ("ham", 0)):
        pool = [t for t, yy in zip(texts_test, y_test) if yy == cls_id]
        for t in pool[:4]:
            examples.append({"label": cls, "text": t[:1500]})

    print(f"  {label} done in {time.time() - t0:.1f}s")
    return {"metrics": metrics, "examples": examples,
            "meta": detector.metadata}


def results_markdown(all_results: dict) -> str:
    lines = ["# Results\n"]
    for modality, res in all_results.items():
        meta = res["meta"]
        lines += [f"## {MODALITY_LABEL[modality]}",
                  f"\n{meta['n_messages']:,} messages "
                  f"({meta['spam_share']:.1%} spam) — {meta['n_train']:,} train / "
                  f"{meta['n_test']:,} test, {meta['n_features']:,} features.\n",
                  "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (5-fold) |",
                  "|---|---|---|---|---|---|---|"]
        for model in MODEL_DISPLAY_ORDER:
            m = res["metrics"][model]
            cv = (f"{m['cv_f1_mean']:.4f} ± {m['cv_f1_std']:.4f}"
                  if "cv_f1_mean" in m else "—")
            star = "**" if model == "Hybrid Ensemble" else ""
            lines.append(f"| {star}{model}{star} | {m['accuracy']:.4f} | "
                         f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                         f"{m['roc_auc']:.4f} | {cv} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    apply_style()
    t0 = time.time()
    dfs = {m: load_processed(m) for m in MODALITIES}
    fig_dataset_overview(dfs)

    all_results = {m: train_modality(m) for m in MODALITIES}

    metrics_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modalities": {m: {"meta": r["meta"], "models": r["metrics"]}
                       for m, r in all_results.items()},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    (REPORTS_DIR / "examples.json").write_text(json.dumps(
        {m: r["examples"] for m, r in all_results.items()}, indent=2))
    (REPORTS_DIR / "results.md").write_text(results_markdown(all_results))

    print(f"\nAll artefacts written to {REPORTS_DIR} and {MODELS_DIR}")
    print(f"Total pipeline time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
