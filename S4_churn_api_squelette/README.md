# Churn Predictor API — Squelette à compléter

## Votre mission
Compléter les fichiers marqués `# TODO`. À la fin de la séance, votre API doit :
- Démarrer avec `uvicorn app.main:app --reload`
- Afficher Swagger sur http://localhost:8000/docs
- Faire passer **tous les tests** : `pytest -v`

## Ordre suggéré
1. `app/schemas.py` — modèles Pydantic
2. `app/model.py` — chargement modèle + prédiction
3. `app/main.py` — endpoints FastAPI
4. Tester avec Swagger
5. Lire et comprendre `tests/test_api.py` puis lancer `pytest`

## Setup
```bash
pip install -r requirements.txt
```

⚠️ Mettez votre `best_model.joblib` et `manifest.json` (issus de S2) dans `artifacts/`.
