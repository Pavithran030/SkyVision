"""Inference pipeline and unified prediction service for SkyVision."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

import config
from src.dataset import validate_image_file, ImageValidationResult
from src.models import load_cloud_checkpoint, load_artifact

CLOUD_DESCRIPTIONS = {
    "clear_sky": "Unobstructed blue sky with virtually no visible cloud cover. Minimal atmospheric scattering and near-zero immediate rain potential.",
    "patterned_clouds": "Layered or rippled clouds (e.g. Altocumulus or Stratocumulus) arranged in waves. Indicates stable mid-level atmospheric moisture.",
    "thick_dark_clouds": "Dense, dark vertical clouds (such as Cumulonimbus or Nimbostratus) causing significant light attenuation. High immediate precipitation probability.",
    "thick_white_clouds": "Bright, high-reflectance cumulus clouds with strong vertical growth. Moderate moisture with variable localized rain risk.",
    "veil_clouds": "High, thin, translucent cirrostratus veil creating a milky sky appearance. Suggests high-altitude ice crystals ahead of warm fronts.",
}


class CloudInferenceEngine:
    """Inference engine for sky pattern classification and feature representation."""

    def __init__(self, checkpoint_path: Path = config.CLOUD_CHECKPOINT, device: str = "cpu"):
        self.device = device
        self.model, self.checkpoint = load_cloud_checkpoint(checkpoint_path, device=device)
        self.idx_to_class = self.checkpoint["idx_to_class"]
        self.transform = T.Compose([
            T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ])

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Predict cloud pattern from PIL image."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            embeddings = self.model.extract_features(tensor).cpu().numpy()[0]

        prob_dict = {self.idx_to_class[i]: float(p) for i, p in enumerate(probs)}
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        top_class = next(iter(sorted_probs))

        return {
            "predicted_class": top_class,
            "confidence": sorted_probs[top_class],
            "probabilities": sorted_probs,
            "explanation": CLOUD_DESCRIPTIONS.get(top_class, "Standard sky morphology category."),
            "features": embeddings,
        }


class PredictionService:
    """Unified service coordinating image validation, vision inference, and nowcasting."""

    def __init__(self):
        self.cloud_engine: Optional[CloudInferenceEngine] = None
        self.rainfall_classifier = None
        self.rainfall_regressor = None
        self.multimodal_model = None
        self.rainfall_metadata: Dict[str, Any] = {}
        self.reload()

    def reload(self):
        """Load or reload model artifacts."""
        if config.CLOUD_CHECKPOINT.exists():
            try:
                self.cloud_engine = CloudInferenceEngine(config.CLOUD_CHECKPOINT)
            except Exception as e:
                print(f"Warning: Cloud checkpoint load error: {e}")

        if config.RAINFALL_CLASSIFIER.exists() and config.RAINFALL_REGRESSOR.exists():
            try:
                self.rainfall_classifier, self.rainfall_metadata = load_artifact(config.RAINFALL_CLASSIFIER)
                self.rainfall_regressor, _ = load_artifact(config.RAINFALL_REGRESSOR)
            except Exception as e:
                print(f"Warning: Rainfall models load error: {e}")

        if config.MULTIMODAL_CLASSIFIER.exists():
            try:
                self.multimodal_model, _ = load_artifact(config.MULTIMODAL_CLASSIFIER)
            except Exception as e:
                print(f"Warning: Multimodal model load error: {e}")

    def analyze_sky_image(
        self,
        image_input: Union[Path, str, bytes, Image.Image],
        weather_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run end-to-end vision and precipitation nowcasting."""
        val_res = validate_image_file(image_input)
        if not val_res.is_valid:
            return {"success": False, "error": val_res.error_message, "validation": val_res}

        if self.cloud_engine is None:
            return {"success": False, "error": "Cloud model checkpoint not found.", "validation": val_res}

        # 1. Vision Analysis (Task A)
        cloud = self.cloud_engine.predict(val_res.image)

        # 2. Rainfall Nowcasting (Task B)
        rain_result = self._predict_rainfall(cloud["features"], weather_features)

        return {
            "success": True,
            "error": None,
            "validation": val_res,
            "cloud_result": {
                "predicted_class": cloud["predicted_class"],
                "confidence": cloud["confidence"],
                "probabilities": cloud["probabilities"],
                "explanation": cloud["explanation"],
            },
            "rainfall_result": rain_result,
        }

    def _predict_rainfall(
        self,
        features: Optional[np.ndarray],
        weather_features: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        if self.rainfall_classifier is None or self.rainfall_regressor is None or features is None:
            return {
                "is_available": False,
                "message": "Rainfall model checkpoint not loaded.",
                "predicted_category": None,
                "predicted_mm": None,
                "horizon": "next_1_hour",
                "model_type": "unavailable",
            }

        feat_2d = features.reshape(1, -1)
        cat_pred = self.rainfall_classifier.predict(feat_2d)[0]
        mm_pred = float(np.clip(self.rainfall_regressor.predict(feat_2d)[0], 0.0, None))

        mm_pred_cat = None
        if self.multimodal_model is not None and weather_features:
            try:
                w_vals = np.array([weather_features.get(k, 0.0) for k in config.MULTIMODAL_FEATURES]).reshape(1, -1)
                combined = np.hstack([feat_2d, w_vals])
                mm_pred_cat = self.multimodal_model.predict(combined)[0]
            except Exception:
                mm_pred_cat = None

        label_source = self.rainfall_metadata.get("label_source", "open_meteo_proxy_reanalysis")
        model_type = "proxy-label experiment" if "proxy" in label_source.lower() else "image-only"

        return {
            "is_available": True,
            "predicted_category": cat_pred,
            "predicted_mm": round(mm_pred, 2),
            "horizon": "next_1_hour",
            "label_source": label_source,
            "model_type": model_type,
            "multimodal_category": mm_pred_cat,
        }
