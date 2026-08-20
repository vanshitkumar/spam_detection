"""Text cleaning / normalisation shared by every model.

Design choice for interpretability: instead of deleting URLs, e-mail
addresses, phone numbers, money amounts and numbers, they are *mapped to
readable placeholder tokens* (``httpaddr``, ``emailaddr``, ``phonenumbr``,
``moneysymb``, ``numbr``).  Those tokens then show up verbatim in SHAP/LIME
explanations, e.g. "presence of httpaddr pushed this message toward SPAM",
which is far more informative than silently dropping the signal.

No stemming/lemmatisation is applied on purpose: keeping surface word forms
makes every explanation directly readable by a non-expert.
"""

from __future__ import annotations

import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|info|co\.uk)\b)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MONEY_RE = re.compile(r"[$£€₹]\s?\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s?(?:dollars|usd|pounds|gbp|euros?|rs)\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?\d[\s\-().]?){7,14}\d\b")
NUM_RE = re.compile(r"\b\d+\b")
HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
WS_RE = re.compile(r"\s+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def normalize_text(text: str) -> str:
    """Lower-case and map high-signal entities to placeholder tokens.

    Used as the ``preprocessor`` of the TF-IDF vectoriser, and re-applied to
    every perturbed sample LIME generates, so explanations always operate on
    exactly what the models see.
    """
    t = str(text).lower()
    t = HTML_TAG_RE.sub(" ", t)
    t = URL_RE.sub(" httpaddr ", t)
    t = EMAIL_RE.sub(" emailaddr ", t)
    t = MONEY_RE.sub(" moneysymb ", t)
    t = PHONE_RE.sub(" phonenumbr ", t)
    t = NUM_RE.sub(" numbr ", t)
    t = NON_ALPHA_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()
