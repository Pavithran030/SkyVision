"""PyTorch Dataset and transform pipelines for cloud sky photographs."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from cloud_rainfall.config import PROJECT_ROOT


def get_cloud_transforms(
    image_size: int = 224,
    is_train: bool = False,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Construct torchvision transform pipeline.
    Uses conservative augmentations that preserve meteorological validity.
    """
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ])


class CloudImageDataset(Dataset):
    """
    PyTorch Dataset loading sky images from a manifest DataFrame.
    """
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform or get_cloud_transforms(is_train=False)

        # Setup class indexing
        if class_to_idx is None:
            unique_classes = sorted(self.df["class_name"].unique())
            self.class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path

        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        label_name = row["class_name"]
        label_idx = self.class_to_idx.get(label_name, 0)
        return tensor, label_idx, str(img_path)
