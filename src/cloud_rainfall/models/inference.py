"""Single-image inference and feature extraction engine."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from cloud_rainfall.config import PROJECT_ROOT
from cloud_rainfall.data.validation import validate_image_file, ImageValidationResult
from cloud_rainfall.models.cloud_classifier import load_cloud_checkpoint, CloudClassifier

# Domain-specific descriptions for cloud categories
CLOUD_DESCRIPTIONS: Dict[str, str] = {
    "clear_sky": "Unobstructed blue sky with virtually no visible cloud cover. Minimal scattering and very low immediate rain potential.",
    "patterned_clouds": "Layered or rippled clouds (e.g., Altocumulus, Stratocumulus, or Cirrocumulus) arranged in wave-like rows or patches. Typically indicates stable mid-level atmospheric layers.",
    "thick_dark_clouds": "Dense, low-luminance clouds (such as Cumulonimbus or Nimbostratus) indicating heavy optical thickness and active condensation. Strongly associated with impending precipitation.",
    "thick_white_clouds": "Bright, high-reflectance clouds (such as fair-weather Cumulus or Altostratus). Substantial water droplet density with variable rain risk depending on vertical growth.",
    "veil_clouds": "High, thin, translucent cloud sheet (Cirrostratus) creating a milky appearance often producing a halo around the sun. Indicates moisture at upper troposphere.",
}


class CloudInferenceEngine:
    """Inference engine for sky pattern recognition and feature extraction."""

    def __init__(self, checkpoint_path: Path | str, device: str = "cpu"):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = PROJECT_ROOT / self.checkpoint_path

        self.model, self.checkpoint = load_cloud_checkpoint(self.checkpoint_path, device=device)
        self.class_to_idx = self.checkpoint["class_to_idx"]
        self.idx_to_class = self.checkpoint["idx_to_class"]
        self.image_size = self.checkpoint.get("image_size", 224)
        mean = self.checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
        std = self.checkpoint.get("normalize_std", [0.229, 0.224, 0.225])

        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def predict(
        self,
        image_input: Union[Path, str, bytes, Image.Image],
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Returns:
            Dictionary containing:
            - is_valid: bool
            - predicted_class: str
            - confidence: float
            - probabilities: Dict[str, float]
            - explanation: str
            - features: np.ndarray (embedding vector)
            - validation: ImageValidationResult
        """
        val_res = validate_image_file(image_input)
        if not val_res.is_valid or val_res.image is None:
            return {
                "is_valid": False,
                "error": val_res.error_message or "Invalid image file",
                "validation": val_res,
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": {},
                "explanation": "",
                "features": None,
            }

        img_tensor = self.transform(val_res.image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            embeddings = self.model.extract_features(img_tensor).cpu().numpy()[0]

        # Map to class names
        prob_dict = {
            self.idx_to_class.get(idx, f"class_{idx}"): float(prob)
            for idx, prob in enumerate(probs)
        }
        # Sort descending
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        top_class = next(iter(sorted_probs))
        top_conf = sorted_probs[top_class]

        explanation = CLOUD_DESCRIPTIONS.get(
            top_class,
            f"Predicted sky pattern category: {top_class}."
        )

        return {
            "is_valid": True,
            "error": None,
            "validation": val_res,
            "predicted_class": top_class,
            "confidence": top_conf,
            "probabilities": sorted_probs,
            "explanation": explanation,
            "features": embeddings,
        }
