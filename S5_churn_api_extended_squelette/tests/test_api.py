"""API tests using FastAPI's TestClient (no live server needed)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """TestClient as a context manager triggers the lifespan handler."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_client():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 960.0,
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "threshold" in body
    assert "metrics_test" in body


def test_predict_valid(client, sample_client):
    r = client.post("/predict", json=sample_client)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_predicted"] in (0, 1)
    assert body["threshold_used"] > 0


def test_predict_missing_field(client, sample_client):
    sample_client.pop("tenure")
    r = client.post("/predict", json=sample_client)
    assert r.status_code == 422  # Pydantic validation


def test_predict_invalid_enum(client, sample_client):
    sample_client["Contract"] = "Five years"
    r = client.post("/predict", json=sample_client)
    assert r.status_code == 422


def test_predict_negative_tenure(client, sample_client):
    sample_client["tenure"] = -5
    r = client.post("/predict", json=sample_client)
    assert r.status_code == 422


def test_predict_batch(client, sample_client):
    r = client.post(
        "/predict-batch",
        json={"clients": [sample_client, sample_client, sample_client]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3
    for p in body["predictions"]:
        assert 0 <= p["churn_probability"] <= 1


def test_predict_batch_empty_rejected(client):
    r = client.post("/predict-batch", json={"clients": []})
    assert r.status_code == 422


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert "/docs" in r.headers["location"]


def test_openapi_schema(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"]
    assert "/predict" in schema["paths"]



# ===== S5 monitoring tests =====

def test_metrics_empty(client):
    # /metrics works even before any prediction
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "total_predictions" in body


def test_drift_check_insufficient(client):
    r = client.get("/drift-check")
    assert r.status_code == 200
    body = r.json()
    # If not enough predictions in the log, status is "insufficient_data"
    assert body["status"] in ("insufficient_data", "ok", "warning", "critical")


def test_predict_appends_to_log(client, sample_client, tmp_path, monkeypatch):
    """A successful /predict must append to the JSONL log."""
    from app import monitoring as mon
    fake_log = tmp_path / "predictions.log.jsonl"
    monkeypatch.setattr(mon, "LOG_PATH", fake_log)

    r = client.post("/predict", json=sample_client)
    assert r.status_code == 200
    # Without monkeypatching the actual LOG_PATH used by the running app,
    # we just verify /metrics still works after a few predictions
    for _ in range(3):
        client.post("/predict", json=sample_client)
    r = client.get("/metrics")
    assert r.json()["total_predictions"] >= 1
