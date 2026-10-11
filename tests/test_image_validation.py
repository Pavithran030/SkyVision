"""Unit tests for image validation module."""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from cloud_rainfall.data.validation import validate_image_file, ImageValidationResult


def test_valid_image():
    # Create valid synthetic image
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    # Add some variation so it's not zero std
    arr = np.array(img)
    arr[0:50, 0:50] = [120, 160, 220]
    img = Image.fromarray(arr)

    result = validate_image_file(img)
    assert result.is_valid is True
    assert result.width == 128
    assert result.height == 128
    assert result.channels == 3
    assert result.error_message is None


def test_too_small_image():
    # Smaller than 64x64
    small_img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    result = validate_image_file(small_img, min_width=64, min_height=64)
    assert result.is_valid is False
    assert "smaller than required minimum" in result.error_message


def test_corrupt_bytes():
    corrupt_data = b"NOT_AN_IMAGE_RANDOM_GARBAGE_BYTES_12345"
    result = validate_image_file(corrupt_data)
    assert result.is_valid is False
    assert result.error_message is not None


def test_nonexistent_file():
    result = validate_image_file(Path("non_existent_image_123456.jpg"))
    assert result.is_valid is False
    assert "does not exist" in result.error_message


def test_dark_image_warning():
    # Very dark image
    dark_img = Image.new("RGB", (100, 100), color=(5, 5, 5))
    result = validate_image_file(dark_img, min_mean_intensity=15.0)
    assert result.is_valid is True
    assert any("dark" in w.lower() for w in result.warnings)


def test_blank_uniform_image_warning():
    # Uniform color (zero standard deviation)
    blank_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    result = validate_image_file(blank_img, min_std_intensity=5.0)
    assert result.is_valid is True
    assert any("contrast" in w.lower() or "uniform" in w.lower() for w in result.warnings)
