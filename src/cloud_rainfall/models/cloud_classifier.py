"""Transfer-learning Cloud Classifier based on MobileNetV3."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    resnet18,
    ResNet18_Weights,
)

from cloud_rainfall.config import PROJECT_ROOT


class CloudClassifier(nn.Module):
    """
    Deep vision classifier with transfer learning and feature extraction.
    """
    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            # MobileNetV3 small classifier head:
            # (0): Linear(576, 1024), (1): Hardswish(), (2): Dropout, (3): Linear(1024, num_classes)
            in_features = self.backbone.classifier[3].in_features
            self.embedding_dim = 1024
            self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
        elif backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.embedding_dim = in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent feature embeddings for downstream rainfall models.
        Returns a 1D feature vector per image in batch.
        """
        if self.backbone_name == "mobilenet_v3_small":
            features = self.backbone.features(x)
            pooled = self.backbone.avgpool(features)
            flattened = torch.flatten(pooled, 1)
            # Extract features before final classification layer
            embedding = self.backbone.classifier[0](flattened)
            embedding = self.backbone.classifier[1](embedding)
            return embedding
        elif self.backbone_name == "resnet18":
            # For resnet18, extract before fc
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)


def build_cloud_classifier(
    num_classes: int = 5,
    backbone_name: str = "mobilenet_v3_small",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> CloudClassifier:
    """Build and configure the cloud classifier."""
    model = CloudClassifier(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=pretrained,
    )
    if freeze_backbone:
        for param in model.backbone.features.parameters():
            param.requires_grad = False
    return model


def save_cloud_checkpoint(
    model: CloudClassifier,
    save_path: Path | str,
    class_to_idx: Dict[str, int],
    image_size: int = 224,
    normalize_mean: List[float] = [0.485, 0.456, 0.406],
    normalize_std: List[float] = [0.229, 0.224, 0.225],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint with all reproduction metadata."""
    path = Path(save_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    checkpoint = {
        "state_dict": model.state_dict(),
        "backbone_name": model.backbone_name,
        "num_classes": model.num_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_size": image_size,
        "normalize_mean": normalize_mean,
        "normalize_std": normalize_std,
        "extra_metadata": extra_metadata or {},
    }
    torch.save(checkpoint, path)


def load_cloud_checkpoint(
    checkpoint_path: Path | str,
    device: str = "cpu",
) -> Tuple[CloudClassifier, Dict[str, Any]]:
    """Load model and metadata from a saved checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    backbone = checkpoint["backbone_name"]

    model = CloudClassifier(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
