"""Configuration: paths and API metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

# Monitoring (S5)
MONITORING_DIR = ROOT / "monitoring"
LOG_PATH = MONITORING_DIR / "predictions.log.jsonl"
BASELINE_TRAIN_PATH = ARTIFACTS_DIR / "baseline_train.csv"  # snapshot of X_train_fe at training time

# PSI thresholds (rule of thumb)
PSI_OK_THRESHOLD = 0.10
PSI_WARNING_THRESHOLD = 0.25

API_TITLE = "Churn Predictor API"
API_VERSION = "1.1.0"  # bumped for S5 monitoring features
API_DESCRIPTION = (
    "Predict telco customer churn from a fiche client. "
    "Includes monitoring & drift detection (S5)."
)
