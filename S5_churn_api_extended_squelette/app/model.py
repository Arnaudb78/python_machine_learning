"""Model loading and prediction service.

🎯 Mission: implement ModelService that loads the model + manifest,
applies feature engineering, and predicts.

⚠️ The feature engineering function MUST stay in sync with the one in S2!
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 4 engineered features built in S2.

    Features to add:
      - tenure_group: bin tenure into "new" (<12), "mid" (12-36), "loyal" (>=36)
      - services_count: count of "Yes" across SERVICE_COLS
      - has_internet: 1 if InternetService != "No" else 0
      - avg_charge_per_month: TotalCharges / max(tenure, 1)
    """
    df = df.copy()

    # tenure_group: MUST match S2 (bins=[-1, 12, 36, 1000], labels new/mid/loyal)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 36, 1000],
        labels=["new", "mid", "loyal"],
    )

    df["services_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["avg_charge_per_month"] = df["TotalCharges"] / np.maximum(df["tenure"], 1)

    return df


class ModelService:
    """Loads the model + manifest, exposes predict_one and predict_batch."""

    def __init__(self, model_path: Path, manifest_path: Path):
        self.model = joblib.load(model_path)
        self.manifest = json.loads(Path(manifest_path).read_text())
        self.threshold = float(self.manifest["threshold"])

    def predict_one(self, features_dict: dict) -> dict:
        """Predict churn for ONE client.

        Returns: {"churn_probability": float, "churn_predicted": int, "threshold_used": float}
        """
        df = pd.DataFrame([features_dict])
        df = add_engineered_features(df)
        proba = float(self.model.predict_proba(df)[0, 1])
        return {
            "churn_probability": proba,
            "churn_predicted": int(proba >= self.threshold),
            "threshold_used": self.threshold,
        }

    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        """Predict churn for N clients in one shot."""
        df = pd.DataFrame(features_list)
        df = add_engineered_features(df)
        probas = self.model.predict_proba(df)[:, 1]
        return [
            {
                "churn_probability": float(proba),
                "churn_predicted": int(proba >= self.threshold),
                "threshold_used": self.threshold,
            }
            for proba in probas
        ]
