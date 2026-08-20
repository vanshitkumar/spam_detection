"""Central configuration: paths, constants and model hyper-parameters."""

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ experiment
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

MODALITIES = ("sms", "email")

# Cap the Enron e-mail corpus (~33k messages) to keep training fast on a
# laptop while preserving the class ratio.  Set to None to use everything.
EMAIL_SAMPLE_SIZE = 12_000

# ------------------------------------------------------- feature settings
TFIDF_PARAMS = {
    "sms":   {"max_features": 3000, "ngram_range": (1, 2), "min_df": 2},
    "email": {"max_features": 5000, "ngram_range": (1, 2), "min_df": 3},
}

# Lexicon used for the SPAM_KEYWORD_COUNT engineered feature.  It is a
# transparent, human-auditable signal (part of the interpretability story).
SPAM_KEYWORDS = [
    "free", "win", "winner", "won", "winning", "cash", "prize", "award",
    "urgent", "claim", "congratulations", "congrats", "offer", "click",
    "subscribe", "guaranteed", "guarantee", "credit", "loan", "lottery",
    "million", "bonus", "discount", "cheap", "deal", "limited", "expires",
    "expire", "alert", "verify", "account", "password", "bank", "refund",
    "investment", "viagra", "pills", "ringtone", "voucher", "gift",
    "txt", "sms", "mobile", "selected", "reward", "unsubscribe",
]

# --------------------------------------------------------------- models
LOGREG_PARAMS = {
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 250,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

MODEL_DISPLAY_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "Hybrid Ensemble"]

# ------------------------------------------------------------------ XAI
SHAP_BACKGROUND_SIZE = 800   # test-set subsample used for global SHAP plots
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 1000
