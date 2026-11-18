# Shrimp Pond Night-Time Oxygen Risk – MLOps Project

**Student:** K. Linu Priyanth | **Reg. No.:** 231FA04C97 | **Section:** 15

## Problem
Predict whether dissolved oxygen may fall below 4 mg/L by 5:00 AM using 8:00 PM pond measurements. The model returns High/Normal risk and an additional-aeration recommendation.

## Project structure
- `data/`: synthetic V1 and rainy-period V2 datasets plus version log
- `src/`: generation, preprocessing, feature store, training and FastAPI
- `feature_store/`: reusable engineered feature file
- `models/`: selected model artifact after training
- `tests/`: data, feature and API tests
- `.github/workflows/`: GitHub Actions CI workflow
- `reports/`: model comparison and report evidence

## Run on Windows PowerShell
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.feature_store
mlflow ui --host 127.0.0.1 --port 5000
```
Open a second terminal in the project folder:
```powershell
python -m src.train
pytest -q
uvicorn src.api:app --reload
```
Open `http://127.0.0.1:8000/docs` for Swagger UI. Test `GET /health` and `POST /predict`.

## MLflow
The training script logs two candidate models and metrics to the `Shrimp-Pond-Oxygen-Risk` experiment, saves the best pipeline to `models/best_model.joblib`, and attempts to register it as `ShrimpPondOxygenRisk`. Run MLflow UI to inspect the local tracking store.

## GitHub
Create a repository, push all project files, and confirm the Actions workflow is green. Do not include `.venv/` or temporary cache files.

## Important note
The datasets are synthetic educational examples. A real farm deployment requires validation against measured pond data, calibration, domain review and an operational response protocol. A model prediction is an early-warning aid, not a replacement for dissolved-oxygen monitoring.
