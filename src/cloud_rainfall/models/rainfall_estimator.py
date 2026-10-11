"""Rainfall category classifier and continuous regression models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

from cloud_rainfall.config import PROJECT_ROOT


class RainfallBaselineModel:
    """Baseline models: Majority class for categories, median for regression."""

    def __init__(self):
        self.classifier = DummyClassifier(strategy="most_frequent")
        self.regressor = DummyRegressor(strategy="median")

    def fit(self, X: np.ndarray, y_cat: np.ndarray, y_mm: np.ndarray):
        self.classifier.fit(X, y_cat)
        self.regressor.fit(X, y_mm)
        return self

    def predict_category(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_amount(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.regressor.predict(X), 0.0, None)


class RainfallEmbeddingModel:
    """
    Image-embedding based model using Random Forest on CNN feature representations.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        random_state: int = 42,
    ):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
        )
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y_cat: np.ndarray, y_mm: np.ndarray):
        self.classifier.fit(X, y_cat)
        self.regressor.fit(X, y_mm)
        self.classes_ = self.classifier.classes_
        return self

    def predict_category(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_category_proba(self, X: np.ndarray) -> Dict[str, float]:
        probs = self.classifier.predict_proba(X)[0]
        return {cls_name: float(p) for cls_name, p in zip(self.classes_, probs)}

    def predict_amount(self, X: np.ndarray) -> np.ndarray:
        # Documented post-processing: clip negative values to 0
        preds = self.regressor.predict(X)
        return np.clip(preds, 0.0, None)


class MultimodalRainfallModel:
    """
    Multimodal model combining CNN latent embeddings with surface meteorological features.
    """

    def __init__(
        self,
        feature_names: List[str],
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X_combined: np.ndarray, y_cat: np.ndarray):
        self.classifier.fit(X_combined, y_cat)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X_combined: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X_combined)

    def predict_proba(self, X_combined: np.ndarray) -> Dict[str, float]:
        probs = self.classifier.predict_proba(X_combined)[0]
        return {cls_name: float(p) for cls_name, p in zip(self.classes_, probs)}


def save_rainfall_artifact(
    model: Any,
    save_path: Path | str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save model and metadata dict using joblib."""
    path = Path(save_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "metadata": metadata or {},
    }
    joblib.dump(payload, path)


def load_rainfall_artifact(
    load_path: Path | str,
) -> Tuple[Any, Dict[str, Any]]:
    """Load model and metadata dict using joblib."""
    path = Path(load_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    payload = joblib.load(path)
    return payload["model"], payload.get("metadata", {})
