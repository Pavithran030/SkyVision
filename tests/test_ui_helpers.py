"""Unit tests for UI helper utilities and PredictionService."""

from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from cloud_rainfall.services.prediction_service import PredictionService
from cloud_rainfall.evaluation.plots import plot_probabilities_bar


def test_prediction_service_fallback_on_corrupt_input():
    service = PredictionService()
    # Pass invalid bytes
    res = service.analyze_sky_image(b"CORRUPTED_BYTES")
    assert res["success"] is False
    assert "Corrupt or unreadable" in res["error"]


def test_plot_probabilities_bar():
    probs = {
        "clear_sky": 0.10,
        "patterned_clouds": 0.20,
        "thick_dark_clouds": 0.70,
    }
    fig = plot_probabilities_bar(probs)
    assert fig is not None
