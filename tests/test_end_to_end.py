"""End-to-end integration test verifying the complete inference pipeline."""

from pathlib import Path
from PIL import Image
import pytest

from cloud_rainfall.config import PROJECT_ROOT
from cloud_rainfall.services.prediction_service import PredictionService


def test_end_to_end_analysis_pipeline():
    service = PredictionService()
    assert service.cloud_engine is not None
    assert service.rainfall_classifier is not None
    assert service.rainfall_regressor is not None

    # Load one of the sample raw images
    sample_images = list((PROJECT_ROOT / "data" / "raw").glob("*/*.jpg"))
    assert len(sample_images) > 0, "No sample images found for integration test"

    img_path = sample_images[0]
    weather_input = {
        "temperature_2m": 27.5,
        "relative_humidity_2m": 80.0,
        "surface_pressure": 1010.0,
        "wind_speed_10m": 12.0,
    }

    result = service.analyze_sky_image(img_path, weather_features=weather_input)

    # Verify response structure
    assert result["success"] is True
    assert result["error"] is None

    cloud = result["cloud_result"]
    assert cloud["predicted_class"] in [
        "clear_sky",
        "patterned_clouds",
        "thick_dark_clouds",
        "thick_white_clouds",
        "veil_clouds",
    ]
    assert 0.0 <= cloud["confidence"] <= 1.0
    assert len(cloud["probabilities"]) == 5
    assert len(cloud["explanation"]) > 0

    rain = result["rainfall_result"]
    assert rain["is_available"] is True
    assert rain["predicted_category"] in ["no_measurable_rain", "light", "moderate", "heavy"]
    assert rain["predicted_mm"] >= 0.0
    assert rain["horizon"] == "next_1_hour"
    assert rain["model_type"] in ["image-only", "proxy-label experiment"]
