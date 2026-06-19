"""FastAPI app — extended with monitoring endpoints (S5).

🎯 Mission: extend the S4 main.py with:
- /metrics  : returns traffic stats (count, churn rate, avg proba)
- /drift-check : compares recent input distributions to baseline (PSI)
- log every prediction via app.monitoring.log_prediction
"""
from contextlib import asynccontextmanager
from statistics import mean

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.config import (
    API_TITLE, API_VERSION, API_DESCRIPTION,
    MODEL_PATH, MANIFEST_PATH,
    PSI_OK_THRESHOLD, PSI_WARNING_THRESHOLD,
)
from app.model import ModelService
from app.schemas import (
    ClientFeatures, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse,
    MetricsResponse, DriftCheckResponse,
)
from app.monitoring import (
    log_prediction, read_recent_logs,
    compute_drift, status_from_max_psi, get_baseline,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = ModelService(MODEL_PATH, MANIFEST_PATH)
    yield


app = FastAPI(
    title=API_TITLE, version=API_VERSION, description=API_DESCRIPTION,
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    return HealthResponse(status="ok", model_loaded=hasattr(app.state, "model_service"))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["service"])
def model_info():
    return ModelInfoResponse(**app.state.model_service.manifest)


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict(client: ClientFeatures):
    try:
        features = client.model_dump()
        result = app.state.model_service.predict_one(features)
        log_prediction(features, result)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["predict"])
def predict_batch(req: BatchPredictionRequest):
    try:
        features_list = [c.model_dump() for c in req.clients]
        results = app.state.model_service.predict_batch(features_list)
        for features, result in zip(features_list, results):
            log_prediction(features, result)
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
def metrics():
    """Recent traffic stats from the JSONL log."""
    logs = read_recent_logs(1000)
    if not logs:
        return MetricsResponse(
            total_predictions=0,
            churn_rate_predicted=0.0,
            avg_churn_probability=0.0,
        )
    predicted = [log["prediction"]["churn_predicted"] for log in logs]
    probas = [log["prediction"]["churn_probability"] for log in logs]
    return MetricsResponse(
        total_predictions=len(logs),
        churn_rate_predicted=mean(predicted),
        avg_churn_probability=mean(probas),
    )


@app.get("/drift-check", response_model=DriftCheckResponse, tags=["monitoring"])
def drift_check(min_samples: int = 50):
    """Compare recent input distributions to baseline (PSI)."""
    thresholds = {"ok": PSI_OK_THRESHOLD, "warning": PSI_WARNING_THRESHOLD}
    logs = read_recent_logs(1000)

    if len(logs) < min_samples:
        return DriftCheckResponse(
            status="insufficient_data",
            n_predictions_analyzed=len(logs),
            drift_scores={},
            thresholds=thresholds,
        )

    recent = pd.DataFrame([log["features"] for log in logs])
    baseline = get_baseline()
    scores = compute_drift(baseline, recent)

    return DriftCheckResponse(
        status=status_from_max_psi(scores),
        n_predictions_analyzed=len(logs),
        drift_scores=scores,
        thresholds=thresholds,
    )
