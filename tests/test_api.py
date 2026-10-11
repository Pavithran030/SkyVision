"""Integration tests for FastAPI application endpoints."""

from starlette.testclient import TestClient
from cloud_rainfall.ui.api import app


def test_api_system_status():
    client = TestClient(app)
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "cloud_model_loaded" in data
    assert "rainfall_model_loaded" in data
    assert data["cloud_model_loaded"] is True
    assert data["rainfall_model_loaded"] is True


def test_api_samples_list():
    client = TestClient(app)
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0


def test_api_evaluation_benchmark():
    client = TestClient(app)
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "cloud_accuracy" in data
    assert data["cloud_accuracy"] > 50.0


def test_api_predict_sample():
    client = TestClient(app)
    # Get a sample first
    samples_res = client.get("/api/samples")
    sample_path = samples_res.json()["samples"][0]["relative_path"]

    response = client.post(
        "/api/predict",
        data={
            "sample_relative_path": sample_path,
            "temperature_2m": "28.0",
            "relative_humidity_2m": "75.0",
            "surface_pressure": "1008.0",
            "wind_speed_10m": "14.0",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "cloud_result" in data
    assert data["cloud_result"]["predicted_class"] is not None
    assert "rainfall_result" in data
    assert data["rainfall_result"]["is_available"] is True


def test_api_serves_html_frontend():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SkyVision" in response.text
