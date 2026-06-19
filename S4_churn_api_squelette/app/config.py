"""Configuration: paths and API metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

API_TITLE = "Churn Predictor API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Predict telco customer churn from a fiche client. "
    "Trained in module ML M2 Tech Lead — Digital Campus."
)
