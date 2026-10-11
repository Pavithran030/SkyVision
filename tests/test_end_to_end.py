"""End-to-end integration test verifying complete analysis pipeline."""

from pathlib import Path
from PIL import Image
import pytest

import config
from src.inference import PredictionService


def test_end_to_end_analysis():
    service = PredictionService()
    assert service.cloud_engine is not None
    assert service.rainfall_classifier is not None

    samples = list(config.SAMPLES_DIR.glob("*/*.jpg"))
    assert len(samples) > 0

    img = Image.open(samples[0])
    res = service.analyze_sky_image(img, weather_features={"temperature_2m": 28.0})

    assert res["success"] is True
    assert res["cloud_result"]["predicted_class"] in config.DEFAULT_CLASSES
    assert res["rainfall_result"]["is_available"] is True
    assert res["rainfall_result"]["predicted_mm"] >= 0.0
