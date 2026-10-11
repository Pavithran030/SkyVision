"""Unit tests for rainfall binning, timestamp alignment, and date-aware splitting."""

import numpy as np
import pandas as pd
import pytest

from cloud_rainfall.data.rainfall_pairing import (
    categorize_rainfall,
    build_date_aware_splits,
    validate_pairing_integrity,
    DEFAULT_BINS,
)


def test_categorize_rainfall_bins():
    assert categorize_rainfall(0.0) == "no_measurable_rain"
    assert categorize_rainfall(0.05) == "no_measurable_rain"
    assert categorize_rainfall(0.1) == "light"
    assert categorize_rainfall(1.8) == "light"
    assert categorize_rainfall(2.5) == "moderate"
    assert categorize_rainfall(7.49) == "moderate"
    assert categorize_rainfall(7.5) == "heavy"
    assert categorize_rainfall(25.0) == "heavy"


def test_missing_rainfall_not_zeroed():
    # Crucial scientific check: missing data must NOT become zero rain
    assert categorize_rainfall(None) is None
    assert categorize_rainfall(np.nan) is None


def test_date_aware_splits_no_date_leakage():
    # Create synthetic records spanning 5 days
    records = []
    dates = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05"]
    for d in dates:
        for i in range(10):  # 10 frames per day
            records.append({
                "date": d,
                "frame": f"{d}_{i}",
                "rain_mm_next_hour": 1.0,
            })
    df = pd.DataFrame(records)

    split_df = build_date_aware_splits(df, date_column="date", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    train_dates = set(split_df[split_df["split"] == "train"]["date"])
    val_dates = set(split_df[split_df["split"] == "val"]["date"])
    test_dates = set(split_df[split_df["split"] == "test"]["date"])

    # Ensure zero overlap between splits
    assert len(train_dates.intersection(val_dates)) == 0
    assert len(train_dates.intersection(test_dates)) == 0
    assert len(val_dates.intersection(test_dates)) == 0


def test_pairing_integrity_validation():
    df = pd.DataFrame([
        {"rain_mm_next_hour": 0.0, "rain_source": "gauge", "target_horizon": "next_1_hour", "split": "train"},
        {"rain_mm_next_hour": 2.5, "rain_source": "gauge", "target_horizon": "next_1_hour", "split": "val"},
        {"rain_mm_next_hour": np.nan, "rain_source": "gauge", "target_horizon": "next_1_hour", "split": "test"},
    ])
    report = validate_pairing_integrity(df)
    assert report["total_records"] == 3
    assert report["valid_records"] == 2
    assert report["missing_records"] == 1
    assert report["has_target_horizon"] is True
