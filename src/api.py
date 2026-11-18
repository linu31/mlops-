"""FastAPI model-serving application."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .preprocessing import ROOT, engineer_features, NUMERIC

app = FastAPI(title="Shrimp Pond Night-Time Oxygen Risk API",
              description="Predicts High/Normal oxygen risk by 5:00 AM from 8:00 PM measurements.",
              version="1.0.0")
MODEL_PATH = ROOT/"models"/"best_model.joblib"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class PondNightInput(BaseModel):
    pond_area_acres: float = Field(gt=0)
    stocking_density_per_m2: float = Field(gt=0)
    do_8pm_mg_l: float = Field(gt=0, le=20)
    water_temp_c: float
    daily_feed_kg: float = Field(ge=0)
    planned_aerator_hours: float = Field(ge=0, le=12)
    expected_rainfall_mm: float = Field(ge=0)

@app.get("/health")
def health():
    return {"status":"healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: PondNightInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python -m src.train first.")
    raw = pd.DataFrame([payload.model_dump() if hasattr(payload,"model_dump") else payload.dict()])
    row = engineer_features(raw)
    prediction = int(model.predict(row[NUMERIC])[0])
    risk = "High" if prediction == 1 else "Normal"
    probability = None
    if hasattr(model, "predict_proba"):
        probability = round(float(model.predict_proba(row[NUMERIC])[0][1]), 4)
    return {
        "oxygen_risk": risk,
        "high_risk_probability": probability,
        "additional_night_aeration_may_be_required": risk == "High",
        "recommendation": ("Consider increasing/ extending aeration and verify dissolved oxygen during the night."
                           if risk == "High" else
                           "Continue planned aeration and routine dissolved-oxygen monitoring.")
    }
