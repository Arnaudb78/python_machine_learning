"""FastAPI app — endpoints and lifespan.

🎯 Mission: wire the routes:
  GET  /          → redirect to /docs
  GET  /health    → liveness probe
  GET  /model-info → manifest passthrough
  POST /predict   → one client
  POST /predict-batch → N clients

Use a `lifespan` context manager to load the model ONCE at startup
(not per request — that would be O(N) joblib loads).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.config import (
    API_TITLE, API_VERSION, API_DESCRIPTION,
    MODEL_PATH, MANIFEST_PATH,
)
from app.model import ModelService
from app.schemas import (
    ClientFeatures, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    app.state.model_service = ModelService(MODEL_PATH, MANIFEST_PATH)
    print(f"✅ Model loaded: {app.state.model_service.manifest.get('model_name')}")
    yield
    # Optional cleanup on shutdown
    app.state.model_service = None


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    """Liveness probe."""
    model_loaded = getattr(app.state, "model_service", None) is not None
    return HealthResponse(
        status="ok" if model_loaded else "error",
        model_loaded=model_loaded,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["service"])
def model_info():
    """Return manifest content."""
    return ModelInfoResponse(**app.state.model_service.manifest)


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict(client: ClientFeatures):
    """Predict churn for ONE client."""
    try:
        result = app.state.model_service.predict_one(client.model_dump())
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["predict"])
def predict_batch(req: BatchPredictionRequest):
    """Predict churn for N clients."""
    try:
        features = [c.model_dump() for c in req.clients]
        results = app.state.model_service.predict_batch(features)
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")
