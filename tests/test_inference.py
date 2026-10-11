"""Unit tests for model architecture, checkpoints, and inference engine."""

from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

from cloud_rainfall.models.cloud_classifier import (
    build_cloud_classifier,
    save_cloud_checkpoint,
    load_cloud_checkpoint,
)
from cloud_rainfall.models.inference import CloudInferenceEngine


def test_model_forward_and_feature_extraction():
    num_classes = 5
    model = build_cloud_classifier(num_classes=num_classes, backbone_name="mobilenet_v3_small", pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
        features = model.extract_features(dummy_input)

    assert logits.shape == (2, num_classes)
    assert features.shape[0] == 2
    assert features.shape[1] > 0


def test_checkpoint_roundtrip_and_inference(tmp_path):
    num_classes = 3
    classes = ["clear_sky", "patterned_clouds", "thick_dark_clouds"]
    class_to_idx = {name: i for i, name in enumerate(classes)}

    model = build_cloud_classifier(num_classes=num_classes, backbone_name="mobilenet_v3_small", pretrained=False)
    ckpt_path = tmp_path / "test_model.pt"

    save_cloud_checkpoint(
        model=model,
        save_path=ckpt_path,
        class_to_idx=class_to_idx,
        image_size=224,
    )

    assert ckpt_path.exists()

    loaded_model, ckpt_dict = load_cloud_checkpoint(ckpt_path)
    assert ckpt_dict["num_classes"] == 3
    assert ckpt_dict["idx_to_class"][0] == "clear_sky"

    # Test inference engine
    engine = CloudInferenceEngine(ckpt_path)
    test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    res = engine.predict(test_img)

    assert res["is_valid"] is True
    assert res["predicted_class"] in classes
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 3
    total_prob = sum(res["probabilities"].values())
    assert pytest.approx(total_prob, rel=1e-3) == 1.0
    assert res["features"] is not None
