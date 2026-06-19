"""Monitoring & drift detection for the Churn API.

🎯 Mission: implement the 4 functions below.

Drift metric: PSI (Population Stability Index).
- PSI < 0.10 : ok
- 0.10 <= PSI < 0.25 : warning (moderate drift)
- PSI >= 0.25 : critical (significant drift)
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config import (
    LOG_PATH, BASELINE_TRAIN_PATH,
    PSI_OK_THRESHOLD, PSI_WARNING_THRESHOLD,
)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]


def log_prediction(features: dict, prediction: dict) -> None:
    """Append one prediction event to LOG_PATH (JSONL format).

    Each line: {"timestamp": ISO8601, "features": {...}, "prediction": {...}}
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": prediction,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def read_recent_logs(n: int = 1000) -> list[dict]:
    """Read the last n entries from the JSONL log (returns [] if file missing)."""
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        lines = [line for line in f if line.strip()]
    return [json.loads(line) for line in lines[-n:]]


def psi_numeric(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI for numeric features.

    Formula: sum over bins of (actual_pct - expected_pct) * log(actual_pct / expected_pct)
    Use np.linspace on expected to define bin edges, then np.histogram for both.
    Replace zeros with a small epsilon (1e-6) to avoid division/log issues.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Bin edges defined on the baseline; widen the extremes so min/max land inside.
    bin_edges = np.linspace(expected.min(), expected.max(), n_bins + 1)
    bin_edges[0] -= 1e-9
    bin_edges[-1] += 1e-9

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)

    eps = 1e-6
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """PSI for categorical features (sum over modalities)."""
    cats = sorted(set(expected.unique()) | set(actual.unique()))
    expected_pct = expected.value_counts(normalize=True).reindex(cats, fill_value=0).values
    actual_pct = actual.value_counts(normalize=True).reindex(cats, fill_value=0).values

    eps = 1e-6
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def compute_drift(baseline: pd.DataFrame, recent: pd.DataFrame) -> dict:
    """Compute PSI per monitored feature. Return dict {feature: psi_value}."""
    scores = {}
    for feat in NUMERIC_FEATURES:
        if feat in baseline.columns and feat in recent.columns:
            scores[feat] = psi_numeric(baseline[feat].values, recent[feat].values)
    for feat in CATEGORICAL_FEATURES:
        if feat in baseline.columns and feat in recent.columns:
            scores[feat] = psi_categorical(baseline[feat], recent[feat])
    return scores


def status_from_max_psi(scores: dict) -> str:
    """Translate the worst PSI into a global status: ok / warning / critical / no_data."""
    if not scores:
        return "no_data"
    max_psi = max(scores.values())
    if max_psi < PSI_OK_THRESHOLD:
        return "ok"
    if max_psi < PSI_WARNING_THRESHOLD:
        return "warning"
    return "critical"


def get_baseline() -> pd.DataFrame:
    """Load the training-time baseline DataFrame."""
    return pd.read_csv(BASELINE_TRAIN_PATH)
