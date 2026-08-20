"""Feature extraction: TF-IDF text features + engineered stylometric features.

The final design matrix is ``[ TF-IDF (sparse) | engineered stats (scaled) ]``.

Engineered features are computed on the *raw* message (before normalisation)
so they capture surface signals such as SHOUTING, punctuation abuse, URLs and
currency symbols.  Their names are UPPER-CASE so they are visually distinct
from lower-case TF-IDF tokens inside SHAP plots.
"""

from __future__ import annotations

import re
import string

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

from .config import SPAM_KEYWORDS, TFIDF_PARAMS
from .preprocessing import EMAIL_RE, MONEY_RE, PHONE_RE, URL_RE, normalize_text

_KEYWORD_RE = re.compile(r"\b(" + "|".join(map(re.escape, SPAM_KEYWORDS)) + r")\b", re.I)
_PUNCT_SET = set(string.punctuation)

ENGINEERED_FEATURE_NAMES = [
    "CHAR_COUNT",
    "WORD_COUNT",
    "AVG_WORD_LEN",
    "DIGIT_RATIO",
    "UPPERCASE_RATIO",
    "ALLCAPS_WORDS",
    "PUNCT_RATIO",
    "NUM_EXCLAMATIONS",
    "NUM_QUESTIONS",
    "NUM_CURRENCY",
    "NUM_URLS",
    "NUM_EMAILS",
    "NUM_PHONES",
    "SPAM_KEYWORDS",
    "SPECIAL_CHAR_RATIO",
]

# Short human-readable descriptions surfaced in the demo app.
ENGINEERED_FEATURE_DESCRIPTIONS = {
    "CHAR_COUNT": "Length of the message in characters",
    "WORD_COUNT": "Number of words",
    "AVG_WORD_LEN": "Average word length",
    "DIGIT_RATIO": "Share of characters that are digits",
    "UPPERCASE_RATIO": "Share of letters written in UPPER CASE",
    "ALLCAPS_WORDS": "Words written fully in capitals (e.g. 'WINNER')",
    "PUNCT_RATIO": "Share of punctuation characters",
    "NUM_EXCLAMATIONS": "Count of '!'",
    "NUM_QUESTIONS": "Count of '?'",
    "NUM_CURRENCY": "Currency symbols / money amounts ($, £, €…)",
    "NUM_URLS": "Links / web addresses detected",
    "NUM_EMAILS": "E-mail addresses detected",
    "NUM_PHONES": "Phone numbers detected",
    "SPAM_KEYWORDS": "Hits from a transparent scam-keyword lexicon",
    "SPECIAL_CHAR_RATIO": "Share of non-alphanumeric characters",
}


def engineered_features(text: str) -> list[float]:
    """15 interpretable stylometric/structural statistics of one raw message."""
    t = str(text)
    n_chars = len(t)
    words = t.split()
    n_words = len(words)
    letters = [c for c in t if c.isalpha()]
    n_letters = len(letters)

    safe_chars = max(n_chars, 1)
    safe_words = max(n_words, 1)
    safe_letters = max(n_letters, 1)

    return [
        float(n_chars),
        float(n_words),
        sum(len(w) for w in words) / safe_words,
        sum(c.isdigit() for c in t) / safe_chars,
        sum(c.isupper() for c in letters) / safe_letters,
        float(sum(1 for w in words if len(w) >= 2 and w.isupper() and w.isalpha())),
        sum(c in _PUNCT_SET for c in t) / safe_chars,
        float(t.count("!")),
        float(t.count("?")),
        float(len(MONEY_RE.findall(t)) + sum(t.count(s) for s in "$£€₹")),
        float(len(URL_RE.findall(t))),
        float(len(EMAIL_RE.findall(t))),
        float(len(PHONE_RE.findall(t))),
        float(len(_KEYWORD_RE.findall(t))),
        sum(not (c.isalnum() or c.isspace()) for c in t) / safe_chars,
    ]


class EngineeredFeatureExtractor(BaseEstimator, TransformerMixin):
    """Stateless transformer: list of raw texts -> (n, 15) float matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([engineered_features(t) for t in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(ENGINEERED_FEATURE_NAMES)


class SpamFeaturizer(BaseEstimator, TransformerMixin):
    """Full feature pipeline: raw texts -> sparse [TF-IDF | scaled engineered].

    The engineered block is MaxAbs-scaled so Logistic Regression sees
    comparable magnitudes; tree models are scale-invariant so this is
    harmless for them.
    """

    def __init__(self, modality: str = "sms"):
        self.modality = modality

    def fit(self, texts, y=None):
        params = TFIDF_PARAMS[self.modality]
        self.tfidf_ = TfidfVectorizer(
            preprocessor=normalize_text,
            sublinear_tf=True,
            strip_accents="unicode",
            **params,
        )
        tfidf_matrix = self.tfidf_.fit_transform(texts)
        self.engineered_ = EngineeredFeatureExtractor()
        eng = self.engineered_.transform(texts)
        self.scaler_ = MaxAbsScaler().fit(eng)
        self.feature_names_ = list(self.tfidf_.get_feature_names_out()) + ENGINEERED_FEATURE_NAMES
        self.n_tfidf_features_ = tfidf_matrix.shape[1]
        return self

    def transform(self, texts):
        texts = list(texts)
        tfidf_matrix = self.tfidf_.transform(texts)
        eng = self.scaler_.transform(self.engineered_.transform(texts))
        return sparse.hstack([tfidf_matrix, sparse.csr_matrix(eng)], format="csr")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_)
