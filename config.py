"""Central configuration and paths for SkyVision."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory Paths (Flat, professional 1-level structure)
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CLOUD_MANIFEST = DATA_DIR / "cloud_manifest.csv"
RAINFALL_MANIFEST = DATA_DIR / "rainfall_manifest.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLOUD_CHECKPOINT = ARTIFACTS_DIR / "cloud_classifier.pt"
RAINFALL_CLASSIFIER = ARTIFACTS_DIR / "rainfall_classifier.joblib"
RAINFALL_REGRESSOR = ARTIFACTS_DIR / "rainfall_regressor.joblib"
MULTIMODAL_CLASSIFIER = ARTIFACTS_DIR / "multimodal_classifier.joblib"
METADATA_FILE = ARTIFACTS_DIR / "metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
STATIC_DIR = PROJECT_ROOT / "static"

# Model & Domain Specifications
DEFAULT_CLASSES = [
    "clear_sky",
    "patterned_clouds",
    "thick_dark_clouds",
    "thick_white_clouds",
    "veil_clouds",
]

IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Rainfall Bins (1-hour accumulated precipitation in mm)
RAINFALL_BINS = [
    {"name": "no_measurable_rain", "min_mm": 0.0, "max_mm": 0.1},
    {"name": "light", "min_mm": 0.1, "max_mm": 2.5},
    {"name": "moderate", "min_mm": 2.5, "max_mm": 7.5},
    {"name": "heavy", "min_mm": 7.5, "max_mm": 9999.0},
]

MULTIMODAL_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
]
