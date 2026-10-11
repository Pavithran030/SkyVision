"""Unit tests for data validation, transforms, and rainfall pairing."""

from PIL import Image
import numpy as np
import pytest

from src.dataset import (
    validate_image_file,
    categorize_rainfall,
    build_date_aware_splits,
    get_cloud_transforms,
)
import pandas as pd


def test_valid_image_validation():
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    res = validate_image_file(img)
    assert res.is_valid is True
    assert res.width == 128
    assert res.height == 128


def test_invalid_image_resolution():
    small_img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    res = validate_image_file(small_img)
    assert res.is_valid is False
    assert "smaller than minimum" in res.error_message


def test_corrupt_data():
    res = validate_image_file(b"CORRUPT_BYTES")
    assert res.is_valid is False


def test_rainfall_binning():
    assert categorize_rainfall(0.0) == "no_measurable_rain"
    assert categorize_rainfall(0.05) == "no_measurable_rain"
    assert categorize_rainfall(0.1) == "light"
    assert categorize_rainfall(2.5) == "moderate"
    assert categorize_rainfall(8.0) == "heavy"
    assert categorize_rainfall(None) is None


def test_date_aware_splits():
    records = [{"date": f"2026-08-{d:02d}", "val": i} for d in range(1, 10) for i in range(5)]
    df = pd.DataFrame(records)
    split_df = build_date_aware_splits(df)

    train_dates = set(split_df[split_df["split"] == "train"]["date"])
    val_dates = set(split_df[split_df["split"] == "val"]["date"])
    test_dates = set(split_df[split_df["split"] == "test"]["date"])

    assert len(train_dates.intersection(val_dates)) == 0
    assert len(train_dates.intersection(test_dates)) == 0
