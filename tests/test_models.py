"""Unit tests for MobileNetV3 CNN, rainfall estimators, and inference."""

from PIL import Image
import numpy as np
import pytest
import torch

import config
from src.models import CloudClassifier, RainfallEmbeddingModel
from src.inference import CloudInferenceEngine


def test_cloud_classifier_forward_and_embedding():
    model = CloudClassifier(num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    features = model.extract_features(x)

    assert logits.shape == (2, 5)
    assert features.shape == (2, 1024)


def test_rainfall_embedding_model_fit_predict():
    X = np.random.randn(20, 1024)
    y_cat = np.array(["light", "heavy"] * 10)
    y_mm = np.random.uniform(0.0, 10.0, 20)

    rf = RainfallEmbeddingModel(n_estimators=10)
    rf.fit(X, y_cat, y_mm)

    cat_pred = rf.predict_category(X[:2])
    mm_pred = rf.predict_amount(X[:2])

    assert len(cat_pred) == 2
    assert len(mm_pred) == 2
    assert (mm_pred >= 0.0).all()


def test_cloud_inference_engine():
    if config.CLOUD_CHECKPOINT.exists():
        engine = CloudInferenceEngine(config.CLOUD_CHECKPOINT)
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        res = engine.predict(img)

        assert res["predicted_class"] in config.DEFAULT_CLASSES
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["probabilities"]) == 5
