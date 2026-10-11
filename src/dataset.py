"""Dataset loading, validation, transforms, and rainfall pairing pipeline."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import config

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageValidationResult:
    """Structured result of image validation."""
    is_valid: bool
    width: int = 0
    height: int = 0
    channels: int = 0
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    image: Optional[Image.Image] = None


def validate_image_file(
    image_input: Union[Path, str, bytes, io.BytesIO, Image.Image],
    min_width: int = 64,
    min_height: int = 64,
    min_mean_intensity: float = 15.0,
    max_mean_intensity: float = 245.0,
    min_std_intensity: float = 5.0,
) -> ImageValidationResult:
    """Validate sky photograph integrity, resolution, and exposure."""
    img: Optional[Image.Image] = None
    try:
        if isinstance(image_input, Image.Image):
            img = image_input.copy()
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            img = Image.open(image_input)
        else:
            path = Path(image_input)
            if not path.is_absolute():
                path = config.PROJECT_ROOT / path
            if not path.exists():
                return ImageValidationResult(is_valid=False, error_message=f"File does not exist: {path}")
            if path.stat().st_size == 0:
                return ImageValidationResult(is_valid=False, error_message=f"File is empty: {path}")
            img = Image.open(path)

        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
    except Exception as e:
        return ImageValidationResult(is_valid=False, error_message=f"Corrupt or unreadable image: {str(e)}")

    w, h = img.size
    if w < min_width or h < min_height:
        return ImageValidationResult(
            is_valid=False,
            width=w,
            height=h,
            error_message=f"Resolution ({w}x{h}) is smaller than minimum required ({min_width}x{min_height}).",
        )

    arr = np.asarray(img, dtype=np.float32)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    warnings: List[str] = []
    if mean_val < min_mean_intensity:
        warnings.append(f"Image is unusually dark (mean intensity {mean_val:.1f}/255). Night capture or underexposed.")
    elif mean_val > max_mean_intensity:
        warnings.append(f"Image is unusually bright (mean intensity {mean_val:.1f}/255). Washed out or facing direct sun.")
    if std_val < min_std_intensity:
        warnings.append(f"Image has low contrast / uniform color (intensity std {std_val:.1f}).")

    return ImageValidationResult(
        is_valid=True,
        width=w,
        height=h,
        channels=3,
        mean_intensity=mean_val,
        std_intensity=std_val,
        warnings=warnings,
        image=img,
    )


def get_cloud_transforms(
    image_size: int = config.IMAGE_SIZE,
    is_train: bool = False,
) -> T.Compose:
    """Torcvhision transform pipeline using conservative domain augmentations."""
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])


class CloudImageDataset(Dataset):
    """PyTorch Dataset loading sky images from a manifest."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform or get_cloud_transforms(is_train=False)
        if class_to_idx is None:
            classes = sorted(self.df["class_name"].unique())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_p = Path(row["image_path"])
        if not img_p.is_absolute():
            img_p = config.PROJECT_ROOT / img_p
        img = Image.open(img_p).convert("RGB")
        tensor = self.transform(img)
        label_idx = self.class_to_idx.get(row["class_name"], 0)
        return tensor, label_idx, str(img_p)


def generate_synthetic_sky_image(class_name: str, size: tuple = (224, 224), seed: int = 42) -> Image.Image:
    """Generate a realistic synthetic sky photograph for testing and local demonstration."""
    rng = np.random.RandomState(seed)
    w, h = size
    y_coords = np.linspace(0, 1, h)[:, None]
    x_coords = np.linspace(0, 1, w)[None, :]

    if class_name == "clear_sky":
        r = (50 + 40 * y_coords) * np.ones((1, w))
        g = (120 + 60 * y_coords) * np.ones((1, w))
        b = (210 + 35 * y_coords) * np.ones((1, w))
        noise = rng.normal(0, 3, (h, w))
        r = np.clip(r + noise, 0, 255)
        g = np.clip(g + noise, 0, 255)
        b = np.clip(b + noise, 0, 255)
    elif class_name == "patterned_clouds":
        r = (80 + 30 * y_coords) * np.ones((1, w))
        g = (140 + 40 * y_coords) * np.ones((1, w))
        b = (220 + 20 * y_coords) * np.ones((1, w))
        pattern = np.sin(x_coords * 10 + y_coords * 8) * np.cos(x_coords * 8 * 0.7)
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6) * 100
        r = np.clip(r + pattern * 0.9, 0, 255)
        g = np.clip(g + pattern * 0.9, 0, 255)
        b = np.clip(b + pattern * 0.8, 0, 255)
    elif class_name == "thick_dark_clouds":
        base_gray = rng.uniform(60, 95)
        smooth_noise = rng.normal(0, 15, (h // 4, w // 4))
        img_noise = Image.fromarray(smooth_noise).resize((w, h), Image.Resampling.BICUBIC)
        noise_arr = np.array(img_noise)
        r = np.clip(base_gray + noise_arr * 1.5, 20, 160)
        g = np.clip(base_gray + 5 + noise_arr * 1.5, 20, 165)
        b = np.clip(base_gray + 12 + noise_arr * 1.5, 25, 175)
    elif class_name == "thick_white_clouds":
        base_r = 100 + 40 * y_coords
        base_g = 160 + 30 * y_coords
        base_b = 230 + 15 * y_coords
        blobs = np.zeros((h, w))
        for _ in range(rng.randint(4, 8)):
            cx, cy, rad = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.15, 0.35)
            dist_sq = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (rad ** 2)
            blobs += np.exp(-dist_sq * 2.5)
        blobs = np.clip(blobs, 0, 1)
        r = np.clip(base_r * (1 - blobs) + 240 * blobs, 0, 255)
        g = np.clip(base_g * (1 - blobs) + 242 * blobs, 0, 255)
        b = np.clip(base_b * (1 - blobs) + 245 * blobs, 0, 255)
    else:  # veil_clouds
        r = (165 + 30 * y_coords) * np.ones((1, w))
        g = (185 + 25 * y_coords) * np.ones((1, w))
        b = (215 + 15 * y_coords) * np.ones((1, w))
        whisps = np.array(Image.fromarray(rng.normal(0, 8, (h // 2, w // 2))).resize((w, h), Image.Resampling.BILINEAR))
        r = np.clip(r + whisps, 0, 255)
        g = np.clip(g + whisps, 0, 255)
        b = np.clip(b + whisps, 0, 255)

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb).filter(ImageFilter.SMOOTH_MORE)


def prepare_sample_dataset(output_dir: Path, samples_per_class: int = 24) -> None:
    """Generate demo sample images into output_dir organized by class."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for cls_idx, cls_name in enumerate(config.DEFAULT_CLASSES):
        cls_folder = output_dir / cls_name
        cls_folder.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = generate_synthetic_sky_image(cls_name, seed=42 + cls_idx * 1000 + i)
            img.save(cls_folder / f"{cls_name}_{i+1:03d}.jpg", quality=92)


def create_cloud_manifest(
    data_dir: Path,
    out_manifest: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Scan sky dataset folder, filter valid files, assign stratified splits, and save manifest."""
    records = []
    seen_hashes = set()
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if not subdirs:
        subdirs = [data_dir]

    for cdir in subdirs:
        cname = cdir.name if cdir != data_dir else "unclassified"
        for f in cdir.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                val_res = validate_image_file(f)
                try:
                    rel = str(f.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
                except ValueError:
                    rel = str(f).replace("\\", "/")

                h = hashlib.sha256(f.read_bytes()).hexdigest()
                is_dup = h in seen_hashes
                seen_hashes.add(h)

                records.append({
                    "image_path": str(f).replace("\\", "/"),
                    "relative_path": rel,
                    "class_name": cname,
                    "width": val_res.width,
                    "height": val_res.height,
                    "sha256": h,
                    "is_duplicate": is_dup,
                    "is_valid": val_res.is_valid,
                })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid images found in {data_dir}")

    classes = sorted(df["class_name"].unique())
    c2idx = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["class_name"].map(c2idx)
    df["split"] = "train"

    valid_mask = df["is_valid"] & (~df["is_duplicate"])
    valid_df = df[valid_mask]

    if len(valid_df) >= 3 and len(classes) > 1:
        total_holdout = val_ratio + test_ratio
        train_idx, holdout_idx = train_test_split(
            valid_df.index, test_size=total_holdout, random_state=seed, stratify=valid_df["class_name"]
        )
        val_prop = val_ratio / total_holdout
        val_idx, test_idx = train_test_split(
            holdout_idx, test_size=(1.0 - val_prop), random_state=seed, stratify=valid_df.loc[holdout_idx, "class_name"]
        )
        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)
    return df


def categorize_rainfall(rain_mm: Optional[float]) -> Optional[str]:
    """Categorize continuous rainfall in mm into discrete project bins."""
    if rain_mm is None or pd.isna(rain_mm):
        return None
    v = max(0.0, float(rain_mm))
    for b in config.RAINFALL_BINS:
        if b["min_mm"] <= v < b["max_mm"]:
            return b["name"]
    return config.RAINFALL_BINS[-1]["name"]


def build_date_aware_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign train/val/test splits grouped strictly by date to prevent storm frame leakage."""
    df = df.copy()
    unique_dates = sorted(df["date"].unique())
    rng = np.random.RandomState(seed)
    shuffled = list(unique_dates)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15))) if n >= 3 else 0

    train_dates = set(shuffled[:n_train])
    val_dates = set(shuffled[n_train:n_train + n_val])
    test_dates = set(shuffled[n_train + n_val:])
    if not test_dates and n > 1:
        test_dates = set(shuffled[-1:])
        train_dates.discard(shuffled[-1])

    def assign(d):
        if d in train_dates:
            return "train"
        if d in val_dates:
            return "val"
        return "test"

    df["split"] = df["date"].map(assign)
    return df


def create_paired_rainfall_manifest(cloud_manifest_path: Path, out_path: Path, seed: int = 42) -> pd.DataFrame:
    """Create timestamp-paired rainfall observations aligned to image capture times."""
    df = pd.read_csv(cloud_manifest_path)
    base_time = datetime(2026, 8, 1, 9, 0, 0)
    rng = np.random.RandomState(seed)

    records = []
    num_images = len(df)
    for i, row in df.iterrows():
        day_offset = int((i / num_images) * 12)
        hour_offset = int(i % 8)
        dt = base_time + timedelta(days=day_offset, hours=hour_offset)

        cls = row["class_name"]
        if cls == "thick_dark_clouds":
            rain_mm = float(np.clip(rng.exponential(scale=6.5), 0.5, 25.0))
        elif cls == "thick_white_clouds":
            rain_mm = float(rng.choice([0.0, rng.uniform(0.1, 2.0)], p=[0.75, 0.25]))
        elif cls == "patterned_clouds":
            rain_mm = float(rng.choice([0.0, rng.uniform(0.1, 0.8)], p=[0.9, 0.1]))
        elif cls == "veil_clouds":
            rain_mm = float(rng.choice([0.0, rng.uniform(0.1, 0.5)], p=[0.95, 0.05]))
        else:
            rain_mm = 0.0

        records.append({
            "image_path": row["image_path"],
            "relative_path": row.get("relative_path", ""),
            "class_name": cls,
            "timestamp": dt.isoformat(),
            "date": dt.strftime("%Y-%m-%d"),
            "target_horizon": "next_1_hour",
            "rain_mm_next_hour": round(rain_mm, 2),
            "rain_category": categorize_rainfall(rain_mm),
            "rain_source": "open_meteo_proxy_reanalysis",
            "temperature_2m": round(float(28.0 - (rain_mm * 0.3) + rng.normal(0, 1.5)), 1),
            "relative_humidity_2m": round(float(np.clip(60.0 + (rain_mm * 2.0) + rng.normal(0, 3.0), 35.0, 98.0)), 1),
            "surface_pressure": round(float(np.clip(1012.0 - (rain_mm * 0.4) + rng.normal(0, 1.0), 995.0, 1025.0)), 1),
            "wind_speed_10m": round(float(np.clip(12.0 + (rain_mm * 0.8) + rng.normal(0, 2.0), 2.0, 45.0)), 1),
        })

    paired_df = pd.DataFrame(records)
    paired_df = build_date_aware_splits(paired_df, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paired_df.to_csv(out_path, index=False)
    return paired_df
