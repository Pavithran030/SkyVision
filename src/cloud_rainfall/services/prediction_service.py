"""Unified prediction service orchestrating vision and rainfall models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from cloud_rainfall.config import PROJECT_ROOT, get_paths_config, get_rainfall_model_config
from cloud_rainfall.models.inference import CloudInferenceEngine
from cloud_rainfall.models.rainfall_estimator import load_rainfall_artifact
from cloud_rainfall.data.validation import validate_image_file


class PredictionService:
    """End-to-end service coordinating image validation, cloud CNN, and rainfall models."""

    def __init__(self):
        self.paths_cfg = get_paths_config()
        self.rain_cfg = get_rainfall_model_config()

        self.cloud_engine: Optional[CloudInferenceEngine] = None
        self.rainfall_classifier = None
        self.rainfall_regressor = None
        self.multimodal_model = None
        self.rainfall_metadata: Dict[str, Any] = {}

        self._load_models()

    def _load_models(self):
        # 1. Cloud model
        cloud_ckpt = PROJECT_ROOT / self.paths_cfg["artifacts"]["cloud_checkpoint"]
        if cloud_ckpt.exists():
            try:
                self.cloud_engine = CloudInferenceEngine(cloud_ckpt)
            except Exception as e:
                print(f"Warning: Failed to load cloud checkpoint: {e}")

        # 2. Rainfall models
        rain_cls_path = PROJECT_ROOT / self.paths_cfg["artifacts"]["rainfall_classifier"]
        rain_reg_path = PROJECT_ROOT / self.paths_cfg["artifacts"]["rainfall_regressor"]
        mm_path = PROJECT_ROOT / self.paths_cfg["artifacts"]["multimodal_classifier"]

        if rain_cls_path.exists() and rain_reg_path.exists():
            try:
                self.rainfall_classifier, self.rainfall_metadata = load_rainfall_artifact(rain_cls_path)
                self.rainfall_regressor, _ = load_rainfall_artifact(rain_reg_path)
            except Exception as e:
                print(f"Warning: Failed to load rainfall models: {e}")

        if mm_path.exists():
            try:
                self.multimodal_model, _ = load_rainfall_artifact(mm_path)
            except Exception as e:
                print(f"Warning: Failed to load multimodal model: {e}")

    def analyze_sky_image(
        self,
        image_input: Union[Path, str, bytes, Image.Image],
        weather_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sky image end-to-end:
        - Image validation
        - Cloud pattern recognition
        - Rainfall estimation (if trained artifacts present)
        """
        val_res = validate_image_file(image_input)
        if not val_res.is_valid:
            return {
                "success": False,
                "error": val_res.error_message,
                "validation": val_res,
                "cloud_result": None,
                "rainfall_result": None,
            }

        if self.cloud_engine is None:
            return {
                "success": False,
                "error": "Cloud classifier model is not loaded or checkpoint file is missing.",
                "validation": val_res,
                "cloud_result": None,
                "rainfall_result": None,
            }

        # 1. Run Cloud CNN
        cloud_res = self.cloud_engine.predict(val_res.image)
        if not cloud_res["is_valid"]:
            return {
                "success": False,
                "error": cloud_res["error"],
                "validation": val_res,
                "cloud_result": None,
                "rainfall_result": None,
            }

        # 2. Rainfall prediction (Mode 2)
        rainfall_res = self._predict_rainfall(cloud_res["features"], weather_features)

        return {
            "success": True,
            "error": None,
            "validation": val_res,
            "cloud_result": {
                "predicted_class": cloud_res["predicted_class"],
                "confidence": cloud_res["confidence"],
                "probabilities": cloud_res["probabilities"],
                "explanation": cloud_res["explanation"],
            },
            "rainfall_result": rainfall_res,
        }

    def _predict_rainfall(
        self,
        features: Optional[np.ndarray],
        weather_features: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Estimate rainfall using image features and optional weather context."""
        if self.rainfall_classifier is None or self.rainfall_regressor is None or features is None:
            return {
                "is_available": False,
                "message": (
                    "Rainfall model is not available or has not been trained on paired data yet. "
                    "In accordance with meteorological standards, rainfall estimates are withheld rather than fabricated."
                ),
                "predicted_category": None,
                "predicted_mm": None,
                "horizon": "next_1_hour",
                "label_source": None,
                "model_type": "unavailable",
                "multimodal_category": None,
            }

        feat_2d = features.reshape(1, -1)
        cat_pred = self.rainfall_classifier.predict(feat_2d)[0]
        mm_pred = float(np.clip(self.rainfall_regressor.predict(feat_2d)[0], 0.0, None))

        # Check multimodal prediction if weather inputs provided
        mm_pred_cat = None
        if self.multimodal_model is not None and weather_features:
            try:
                met_keys = self.rain_cfg["features"]["multimodal"]
                met_vals = np.array([weather_features.get(k, 0.0) for k in met_keys]).reshape(1, -1)
                combined = np.hstack([feat_2d, met_vals])
                mm_pred_cat = self.multimodal_model.predict(combined)[0]
            except Exception:
                mm_pred_cat = None

        label_source = self.rainfall_metadata.get("label_source", "proxy_dataset")
        model_type = "proxy-label experiment" if "proxy" in label_source.lower() else "image-only"

        return {
            "is_available": True,
            "message": "Rainfall estimate computed using image feature representation.",
            "predicted_category": cat_pred,
            "predicted_mm": round(mm_pred, 2),
            "horizon": "next_1_hour",
            "label_source": label_source,
            "model_type": model_type,
            "multimodal_category": mm_pred_cat,
        }
