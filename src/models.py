"""Model layer: three heterogeneous base learners + hybrid soft-voting ensemble.

* Logistic Regression — linear, well-calibrated, strong TF-IDF baseline.
* Random Forest       — bagged trees, robust to noise, non-linear.
* XGBoost             — boosted trees, usually the single best learner and
                        the vehicle for exact TreeSHAP explanations.

The ensemble averages the three predicted spam probabilities using weights
proportional to each model's cross-validated F1 score ("performance-weighted
soft voting"), so a stronger learner gets a proportionally louder vote.
"""

from __future__ import annotations

from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import LOGREG_PARAMS, RF_PARAMS, XGB_PARAMS
from .features import SpamFeaturizer

LABELS = ("ham", "spam")


def build_base_models(scale_pos_weight: float = 1.0) -> dict:
    """Fresh, unfitted base learners. ``scale_pos_weight`` handles class
    imbalance for XGBoost (LR/RF use class_weight='balanced')."""
    return {
        "Logistic Regression": LogisticRegression(**LOGREG_PARAMS),
        "Random Forest": RandomForestClassifier(**RF_PARAMS),
        "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS),
    }


class HybridVotingEnsemble:
    """Weighted soft-voting over *pre-fitted* probabilistic classifiers."""

    def __init__(self, models: dict, weights: dict):
        self.models = models
        total = sum(weights.values())
        self.weights = {name: w / total for name, w in weights.items()}

    def predict_proba(self, X) -> np.ndarray:
        proba = np.zeros((X.shape[0], 2))
        for name, model in self.models.items():
            proba += self.weights[name] * model.predict_proba(X)
        return proba

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SpamDetector:
    """Bundle of featurizer + base models + ensemble for one modality.

    This is the single object the demo app loads: it maps *raw text* to
    predictions, exposes per-model probabilities, and provides the
    ``predict_proba_texts`` callable that LIME perturbs.
    """

    def __init__(self, modality: str, featurizer: SpamFeaturizer,
                 base_models: dict, ensemble: HybridVotingEnsemble,
                 metadata: dict | None = None):
        self.modality = modality
        self.featurizer = featurizer
        self.base_models = base_models
        self.ensemble = ensemble
        self.metadata = metadata or {}
        self.metadata.setdefault("trained_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # ------------------------------------------------------------- helpers
    @property
    def feature_names(self) -> list[str]:
        return self.featurizer.feature_names_

    def transform(self, texts) -> "scipy.sparse.csr_matrix":  # noqa: F821
        return self.featurizer.transform(texts)

    # ---------------------------------------------------------- prediction
    def predict_proba_texts(self, texts) -> np.ndarray:
        """(n, 2) ensemble probabilities from raw texts — LIME's classifier_fn."""
        if isinstance(texts, str):
            texts = [texts]
        return self.ensemble.predict_proba(self.transform(list(texts)))

    def analyse(self, text: str) -> dict:
        """Full breakdown for one message: verdict + every model's probability."""
        X = self.transform([text])
        per_model = {name: float(m.predict_proba(X)[0, 1]) for name, m in self.base_models.items()}
        spam_prob = float(self.ensemble.predict_proba(X)[0, 1])
        return {
            "label": LABELS[int(spam_prob >= 0.5)],
            "spam_probability": spam_prob,
            "per_model": per_model,
            "ensemble_weights": dict(self.ensemble.weights),
        }

    # --------------------------------------------------------- persistence
    def save(self, path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "SpamDetector":
        return joblib.load(path)
