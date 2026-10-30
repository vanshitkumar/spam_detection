"""Regenerate the report's SHAP figures with faithful attributions.

XGBoost treats entries that are absent from a sparse matrix as *missing*, so
SHAP values must be computed on exactly the sparse matrix the model scores;
explaining a densified copy of the input describes a different function.  This
script therefore uses XGBoost's native, exact TreeSHAP (``pred_contribs``) on
the sparse features and renders, in the style of src/train.py:

  figures/sms_shap_beeswarm.png, figures/email_shap_beeswarm.png
      global summaries over the same 800 held-out messages as src/train.py
  figures/sms_shap_waterfall.png
      local explanation of the exhibit SMS used in Chapter 4

Run from the repository root after ``python -m src.train``:
    python report/scripts/make_shap_figures.py [--models-dir models]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIGURES = Path(__file__).resolve().parents[1] / "figures"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, SHAP_BACKGROUND_SIZE, TEST_SIZE
from src.data_loader import load_processed
from src.models import SpamDetector
from src.plotting import INK, INK_2, SURFACE, apply_style, save_fig

MODALITY_LABEL = {"sms": "SMS", "email": "Email"}
EXHIBIT = ("URGENT! You have WON a £2,000 cash prize GUARANTEED. "
           "Call 09061701461 now or visit www.claim-your-prize.com to claim!")


def faithful_shap(detector: SpamDetector, texts: list[str]) -> shap.Explanation:
    """Exact TreeSHAP of the XGBoost component on the sparse matrix it scores."""
    X = detector.transform(texts)
    contrib = detector.base_models["XGBoost"].get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True)
    return shap.Explanation(values=contrib[:, :-1], base_values=contrib[:, -1],
                            data=X.toarray(), feature_names=detector.feature_names)


def held_out_texts(modality: str) -> list[str]:
    """The test split of src/train.py (stratified 80/20, same seed)."""
    df = load_processed(modality)
    texts = df["text"].tolist()
    y = (df["label"] == "spam").to_numpy(dtype=int)
    _, idx_test = train_test_split(np.arange(len(texts)), test_size=TEST_SIZE,
                                   stratify=y, random_state=RANDOM_STATE)
    return [texts[i] for i in idx_test]


def beeswarm(modality: str, detector: SpamDetector) -> None:
    texts = held_out_texts(modality)
    rng = np.random.default_rng(RANDOM_STATE)          # same subset as src/train.py
    idx = rng.choice(len(texts), size=min(SHAP_BACKGROUND_SIZE, len(texts)), replace=False)
    exp = faithful_shap(detector, [texts[i] for i in idx])
    plt.figure()
    shap.plots.beeswarm(exp, max_display=14, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.4, 5.6)
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.xaxis.label.set_color(INK_2)
    fig.suptitle(f"{MODALITY_LABEL[modality]} — SHAP beeswarm "
                 f"(XGBoost component, {len(idx)} test messages)",
                 x=0.02, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, FIGURES / f"{modality}_shap_beeswarm.png")


def waterfall(detector: SpamDetector) -> None:
    plt.figure()
    shap.plots.waterfall(faithful_shap(detector, [EXHIBIT])[0], max_display=12, show=False)
    fig = plt.gcf()          # keep SHAP's own size: resizing afterwards clips in-bar labels
    fig.tight_layout()
    save_fig(fig, FIGURES / "sms_shap_waterfall.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the report's SHAP figures.")
    parser.add_argument("--models-dir", type=Path, default=REPO / "models",
                        help="folder holding {sms,email}_detector.joblib (default: models/)")
    args = parser.parse_args()
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    detectors = {}
    for modality in MODALITY_LABEL:
        path = args.models_dir / f"{modality}_detector.joblib"
        if not path.exists():
            sys.exit(f"{path} not found: run `python -m src.train` first")
        detectors[modality] = SpamDetector.load(path)
        beeswarm(modality, detectors[modality])
    waterfall(detectors["sms"])
    print("SHAP figures written to", FIGURES)


if __name__ == "__main__":
    main()
