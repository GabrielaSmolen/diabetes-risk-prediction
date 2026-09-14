import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    """Creates a TestClient instance wrapped in lifespan context manager.
    
    Using 'with TestClient(app)' guarantees that startup/shutdown 
    lifespan events run and load the ML model into memory.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient):
    """Tests the /health endpoint status and model loading state."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == "True"


def test_predict_success(client: TestClient):
    """Tests /predict endpoint with valid patient data."""
    payload = {
        "pregnancies": 2,
        "glucose": 120,
        "blood_pressure": 70,
        "skin_thickness": 20,
        "insulin": 79,
        "bmi": 25.0,
        "diabetes_pedigree": 0.35,
        "age": 29,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_data(client: TestClient):
    """Tests /predict validation logic with invalid payload (negative age)."""
    payload = {
        "pregnancies": 2,
        "glucose": 120,
        "blood_pressure": 70,
        "skin_thickness": 20,
        "insulin": 79,
        "bmi": 25.0,
        "diabetes_pedigree": 0.35,
        "age": -5,  # Invalid: age cannot be negative
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity