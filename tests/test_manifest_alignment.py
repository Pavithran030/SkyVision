"""Unit tests for dataset manifest creation and validation."""

from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from cloud_rainfall.data.manifests import (
    scan_cloud_dataset_folder,
    create_cloud_manifest,
    load_and_validate_manifest,
)


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Fixture that creates a temporary toy dataset with 2 classes."""
    data_dir = tmp_path / "toy_sky_dataset"
    data_dir.mkdir()

    classes = ["clear_sky", "thick_dark_clouds"]
    for cls in classes:
        cls_dir = data_dir / cls
        cls_dir.mkdir()
        for i in range(5):
            # Create a simple image with subtle variation
            img = Image.new("RGB", (80, 80), color=(100, 150, 200))
            arr = np.array(img)
            arr[i * 5:(i + 1) * 5, :] = [110, 160, 210]
            img = Image.fromarray(arr)
            img.save(cls_dir / f"img_{i}.jpg")

    return data_dir


def test_scan_dataset(temp_dataset_dir):
    df = scan_cloud_dataset_folder(temp_dataset_dir)
    assert len(df) == 10
    assert set(df["class_name"].unique()) == {"clear_sky", "thick_dark_clouds"}
    assert df["is_valid"].all()


def test_create_and_load_manifest(temp_dataset_dir, tmp_path):
    manifest_csv = tmp_path / "test_manifest.csv"
    df = create_cloud_manifest(temp_dataset_dir, manifest_csv, train_split=0.6, val_split=0.2, test_split=0.2)

    assert manifest_csv.exists()
    assert len(df) == 10
    assert "split" in df.columns
    assert "label_id" in df.columns

    loaded_df, summary = load_and_validate_manifest(manifest_csv)
    assert summary["total_images"] == 10
    assert summary["valid_images"] == 10
    assert summary["invalid_images"] == 0
    assert len(summary["classes"]) == 2
