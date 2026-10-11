"""Neural network vision architectures and tabular rainfall estimators."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import config


class CloudClassifier(nn.Module):
    """Deep transfer-learning vision classifier based on MobileNetV3-Small."""

    def __init__(self, num_classes: int = len(config.DEFAULT_CLASSES), pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        # Replace classification head
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
        self.embedding_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1024-dim bottleneck latent embeddings for downstream nowcasting."""
        features = self.backbone.features(x)
        pooled = self.backbone.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        emb = self.backbone.classifier[0](flattened)
        emb = self.backbone.classifier[1](emb)
        return emb


class RainfallBaselineModel:
    """Naive benchmark: Majority class for category, median for rainfall volume."""

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
    """Random Forest nowcasting models trained on 1024-dim CNN embeddings."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, class_weight="balanced"
        )
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y_cat: np.ndarray, y_mm: np.ndarray):
        self.classifier.fit(X, y_cat)
        self.regressor.fit(X, y_mm)
        self.classes_ = self.classifier.classes_
        return self

    def predict_category(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_amount(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.regressor.predict(X), 0.0, None)


class MultimodalRainfallModel:
    """Multimodal model combining visual embeddings with surface weather features."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, class_weight="balanced"
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X_combined: np.ndarray, y_cat: np.ndarray):
        self.classifier.fit(X_combined, y_cat)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X_combined: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X_combined)


def save_cloud_checkpoint(model: CloudClassifier, save_path: Path, class_to_idx: Dict[str, int]) -> None:
    """Save PyTorch weights and class mappings."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "num_classes": model.num_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": {idx: name for name, idx in class_to_idx.items()},
    }
    torch.save(payload, save_path)


def load_cloud_checkpoint(checkpoint_path: Path, device: str = "cpu") -> Tuple[CloudClassifier, Dict[str, Any]]:
    """Load PyTorch model and class mapping."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CloudClassifier(num_classes=data["num_classes"], pretrained=False)
    model.load_state_dict(data["state_dict"])
    model.to(device)
    model.eval()
    return model, data


def save_artifact(model: Any, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save Scikit-learn model and metadata via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata or {}}, path)


def load_artifact(path: Path) -> Tuple[Any, Dict[str, Any]]:
    """Load Scikit-learn model and metadata via joblib."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    payload = joblib.load(path)
    return payload["model"], payload.get("metadata", {})
