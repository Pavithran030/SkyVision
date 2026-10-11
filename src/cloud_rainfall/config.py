"""Configuration loader and management module."""

from pathlib import Path
from typing import Any, Dict
import yaml

# Determine repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Load a YAML file safely."""
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_paths_config() -> Dict[str, Any]:
    """Get project paths configuration."""
    config_file = PROJECT_ROOT / "configs" / "paths.yaml"
    if config_file.exists():
        return load_yaml(config_file)
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "manifests_dir": "data/manifests",
            "cloud_manifest": "data/manifests/cloud_manifest.csv",
            "rainfall_manifest": "data/manifests/rainfall_manifest.csv",
        },
        "artifacts": {
            "dir": "artifacts",
            "cloud_checkpoint": "artifacts/cloud_classifier_best.pt",
            "rainfall_classifier": "artifacts/rainfall_classifier_best.joblib",
            "rainfall_regressor": "artifacts/rainfall_regressor_best.joblib",
            "multimodal_classifier": "artifacts/multimodal_classifier_best.joblib",
            "metadata": "artifacts/model_metadata.json",
        },
        "reports": {
            "dir": "reports",
            "figures_dir": "reports/figures",
            "experiment_log": "reports/experiment_log.md",
            "model_card": "reports/model_card.md",
        },
    }


def get_cloud_model_config() -> Dict[str, Any]:
    """Get cloud model configuration."""
    config_file = PROJECT_ROOT / "configs" / "cloud_model.yaml"
    if config_file.exists():
        return load_yaml(config_file)
    return {
        "model": {
            "backbone": "mobilenet_v3_small",
            "pretrained": True,
            "image_size": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "default_classes": [
                "clear_sky",
                "patterned_clouds",
                "thick_dark_clouds",
                "thick_white_clouds",
                "veil_clouds",
            ],
        },
        "training": {
            "batch_size": 16,
            "epochs": 5,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "random_seed": 42,
            "num_workers": 0,
            "train_split": 0.70,
            "val_split": 0.15,
            "test_split": 0.15,
        },
        "validation_checks": {
            "min_width": 64,
            "min_height": 64,
            "min_mean_intensity": 15.0,
            "max_mean_intensity": 245.0,
        },
    }


def get_rainfall_model_config() -> Dict[str, Any]:
    """Get rainfall model configuration."""
    config_file = PROJECT_ROOT / "configs" / "rainfall_model.yaml"
    if config_file.exists():
        return load_yaml(config_file)
    return {
        "prediction_horizon": "next_1_hour",
        "bins": [
            {"name": "no_measurable_rain", "min_mm": 0.0, "max_mm": 0.1},
            {"name": "light", "min_mm": 0.1, "max_mm": 2.5},
            {"name": "moderate", "min_mm": 2.5, "max_mm": 7.5},
            {"name": "heavy", "min_mm": 7.5, "max_mm": 9999.0},
        ],
        "features": {
            "multimodal": [
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
            ]
        },
        "evaluation": {
            "random_seed": 42,
            "train_split": 0.70,
            "val_split": 0.15,
            "test_split": 0.15,
        },
    }
