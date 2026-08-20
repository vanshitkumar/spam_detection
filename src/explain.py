"""Explainable-AI layer: LIME and SHAP explanations for spam predictions.

Two complementary techniques, applied at two scopes:

* **LIME** (local, model-agnostic) explains the *full hybrid ensemble*: it
  perturbs the raw message (dropping words), re-runs the whole
  featurise→ensemble pipeline on every perturbation, and fits a local linear
  surrogate.  Output: signed word weights for THIS prediction.

* **SHAP** (TreeExplainer, exact and fast) explains the *XGBoost component*
  on the full feature space (TF-IDF tokens + engineered stats), both locally
  (waterfall for one message) and globally (beeswarm/bar over the test set).
  Tree SHAP values are additive in log-odds space and computed exactly —
  no sampling error — which is why the tree component is the vehicle for
  global analysis.
"""

from __future__ import annotations

import numpy as np
import shap
from lime.lime_text import LimeTextExplainer

from .config import LIME_NUM_FEATURES, LIME_NUM_SAMPLES, RANDOM_STATE
from .models import SpamDetector

CLASS_NAMES = ["ham", "spam"]


# ------------------------------------------------------------------- LIME
def lime_explain(detector: SpamDetector, text: str,
                 num_features: int = LIME_NUM_FEATURES,
                 num_samples: int = LIME_NUM_SAMPLES):
    """LIME explanation of the hybrid ensemble for one raw message.

    Returns the LimeTextExplanation object; ``.as_list()`` gives
    ``[(word, weight), …]`` where weight > 0 pushes toward *spam*.
    """
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=RANDOM_STATE)
    return explainer.explain_instance(
        str(text),
        detector.predict_proba_texts,
        num_features=num_features,
        num_samples=num_samples,
        labels=(1,),
    )


# ------------------------------------------------------------------- SHAP
def build_shap_explainer(detector: SpamDetector) -> shap.TreeExplainer:
    """Exact TreeSHAP explainer over the XGBoost base learner (log-odds)."""
    model = detector.base_models["XGBoost"]
    try:
        return shap.TreeExplainer(model)
    except Exception:  # older/newer xgboost-shap interop: fall back to the raw booster
        return shap.TreeExplainer(model.get_booster())


def shap_explain_text(detector: SpamDetector, explainer: shap.TreeExplainer,
                      text: str) -> shap.Explanation:
    """Single-message SHAP explanation (1 row, all features, dense)."""
    row = np.asarray(detector.transform([text]).todense())
    exp = explainer(row, check_additivity=False)
    exp.feature_names = detector.feature_names
    return exp[0]


def shap_explain_matrix(detector: SpamDetector, explainer: shap.TreeExplainer,
                        texts: list[str]) -> shap.Explanation:
    """Batch SHAP explanations used for global summary plots."""
    X = np.asarray(detector.transform(texts).todense())
    exp = explainer(X, check_additivity=False)
    exp.feature_names = detector.feature_names
    return exp


def top_shap_features(exp: shap.Explanation, k: int = 12) -> list[tuple[str, float, float]]:
    """Top-k features of a single-row explanation as (name, shap_value, feature_value)."""
    order = np.argsort(-np.abs(exp.values))[:k]
    return [(exp.feature_names[i], float(exp.values[i]), float(exp.data[i])) for i in order]
