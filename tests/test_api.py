"""Unit tests for FastAPI root application endpoints."""

from starlette.testclient import TestClient
from app import app


def test_api_system_status():
    client = TestClient(app)
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "cloud_model_loaded" in data
    assert "rainfall_model_loaded" in data


def test_api_samples_list():
    client = TestClient(app)
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data


def test_api_serves_html():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "SkyVision" in response.text
