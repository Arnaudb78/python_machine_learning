# Churn Predictor API — S5 squelette

Étendre le repo S4 avec **monitoring** et **drift detection**.

## Votre mission

Compléter les TODOs dans :
1. `app/monitoring.py` — log_prediction, read_recent_logs, psi_numeric, psi_categorical, compute_drift, status_from_max_psi
2. `app/schemas.py` — MetricsResponse, DriftCheckResponse
3. `app/main.py` — endpoints /metrics et /drift-check + appel à log_prediction dans /predict

## Setup

```bash
pip install -r requirements.txt
```

Vérifier que `artifacts/` contient `best_model.joblib`, `manifest.json`, `baseline_train.csv`.

## Lancer

```bash
uvicorn app.main:app --reload
```

## Tester

```bash
pytest -v
```
