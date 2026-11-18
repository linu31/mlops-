from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint_when_model_available():
    health = client.get("/health").json()
    if not health["model_loaded"]:
        return
    payload = {
        "pond_area_acres": 1.2, "stocking_density_per_m2": 115,
        "do_8pm_mg_l": 5.5, "water_temp_c": 30.5,
        "daily_feed_kg": 0.48, "planned_aerator_hours": 3,
        "expected_rainfall_mm": 18
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["oxygen_risk"] in ["High","Normal"]
    assert "additional_night_aeration_may_be_required" in body
