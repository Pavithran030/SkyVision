"""Script to prepare cloud dataset, download or generate demo sky images, and create manifests."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

from cloud_rainfall.config import PROJECT_ROOT, get_paths_config, get_cloud_model_config
from cloud_rainfall.data.manifests import create_cloud_manifest


def generate_synthetic_sky_image(class_name: str, size: tuple = (224, 224), seed: int = 42) -> Image.Image:
    """
    Generate a realistic synthetic sky photograph for testing and local demonstration.
    Classes:
      - clear_sky: Deep azure blue to light cyan atmospheric gradient.
      - patterned_clouds: Cirrocumulus / altocumulus ripples and patches on blue sky.
      - thick_dark_clouds: Low-luminance cumulonimbus / nimbostratus dark gray overcast.
      - thick_white_clouds: Bright cumulus fluffy clouds with high luminance.
      - veil_clouds: Cirrostratus thin milky veil covering the sun/sky.
    """
    rng = np.random.RandomState(seed)
    w, h = size

    # Create vertical sky gradient
    y_coords = np.linspace(0, 1, h)[:, None]
    x_coords = np.linspace(0, 1, w)[None, :]

    if class_name == "clear_sky":
        # Deep blue top to lighter cyan blue bottom
        r = (50 + 40 * y_coords) * np.ones((1, w))
        g = (120 + 60 * y_coords) * np.ones((1, w))
        b = (210 + 35 * y_coords) * np.ones((1, w))
        # Subtle atmospheric noise
        noise = rng.normal(0, 3, (h, w))
        r = np.clip(r + noise, 0, 255)
        g = np.clip(g + noise, 0, 255)
        b = np.clip(b + noise, 0, 255)

    elif class_name == "patterned_clouds":
        # Sky background
        r = (80 + 30 * y_coords) * np.ones((1, w))
        g = (140 + 40 * y_coords) * np.ones((1, w))
        b = (220 + 20 * y_coords) * np.ones((1, w))
        # Periodic ripple cloud pattern
        freq_x = rng.uniform(8, 14)
        freq_y = rng.uniform(6, 12)
        pattern = np.sin(x_coords * freq_x + y_coords * freq_y) * np.cos(x_coords * freq_y * 0.7)
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)
        pattern = pattern * 100
        r = np.clip(r + pattern * 0.9, 0, 255)
        g = np.clip(g + pattern * 0.9, 0, 255)
        b = np.clip(b + pattern * 0.8, 0, 255)

    elif class_name == "thick_dark_clouds":
        # Dark storm clouds (nimbostratus/cumulonimbus), low luminance, turbulent textures
        base_gray = rng.uniform(60, 95)
        smooth_noise = rng.normal(0, 15, (h // 4, w // 4))
        # Resize smooth noise
        img_noise = Image.fromarray(smooth_noise).resize((w, h), Image.Resampling.BICUBIC)
        noise_arr = np.array(img_noise)
        r = np.clip(base_gray + noise_arr * 1.5, 20, 160)
        g = np.clip(base_gray + 5 + noise_arr * 1.5, 20, 165)
        b = np.clip(base_gray + 12 + noise_arr * 1.5, 25, 175)

    elif class_name == "thick_white_clouds":
        # Bright puffy cumulus clouds with intense white centres
        base_r = 100 + 40 * y_coords
        base_g = 160 + 30 * y_coords
        base_b = 230 + 15 * y_coords
        # Cloud blobs
        blobs = np.zeros((h, w))
        num_blobs = rng.randint(4, 9)
        for _ in range(num_blobs):
            cx = rng.uniform(0.2, 0.8)
            cy = rng.uniform(0.2, 0.8)
            radius = rng.uniform(0.15, 0.35)
            dist_sq = ((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (radius ** 2)
            blobs += np.exp(-dist_sq * 2.5)
        blobs = np.clip(blobs, 0, 1)
        r = np.clip(base_r * (1 - blobs) + 240 * blobs, 0, 255)
        g = np.clip(base_g * (1 - blobs) + 242 * blobs, 0, 255)
        b = np.clip(base_b * (1 - blobs) + 245 * blobs, 0, 255)

    elif class_name == "veil_clouds":
        # Cirrostratus hazy milky overcast
        r = (165 + 30 * y_coords) * np.ones((1, w))
        g = (185 + 25 * y_coords) * np.ones((1, w))
        b = (215 + 15 * y_coords) * np.ones((1, w))
        whisps = rng.normal(0, 8, (h // 2, w // 2))
        whisps = np.array(Image.fromarray(whisps).resize((w, h), Image.Resampling.BILINEAR))
        r = np.clip(r + whisps, 0, 255)
        g = np.clip(g + whisps, 0, 255)
        b = np.clip(b + whisps, 0, 255)

    else:
        # Fallback neutral sky
        r = 140 * np.ones((h, w))
        g = 170 * np.ones((h, w))
        b = 210 * np.ones((h, w))

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb)
    return img.filter(ImageFilter.SMOOTH_MORE)


def prepare_sample_dataset(
    output_dir: Path,
    samples_per_class: int = 24,
    random_seed: int = 42,
):
    """Create a balanced sample dataset across the 5 standard sky categories."""
    cloud_cfg = get_cloud_model_config()
    classes = cloud_cfg["model"]["default_classes"]

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating local sample dataset at: {output_dir}")

    total_created = 0
    for cls_idx, cls_name in enumerate(classes):
        cls_folder = output_dir / cls_name
        cls_folder.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            seed = random_seed + cls_idx * 1000 + i
            img = generate_synthetic_sky_image(cls_name, size=(224, 224), seed=seed)
            img_path = cls_folder / f"{cls_name}_{i+1:03d}.jpg"
            img.save(img_path, format="JPEG", quality=92)
            total_created += 1

    print(f"Created {total_created} sample images ({samples_per_class} per class across {len(classes)} classes).")


def main():
    parser = argparse.ArgumentParser(description="Prepare cloud dataset and generate manifest.")
    parser.add_argument("--create-sample", action="store_true", help="Generate local sample sky dataset for testing & demo.")
    parser.add_argument("--samples-per-class", type=int, default=24, help="Images per class for sample generation.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory to read or create raw data.")
    args = parser.parse_args()

    paths_cfg = get_paths_config()
    raw_dir = Path(args.data_dir or paths_cfg["data"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir

    if args.create_sample:
        prepare_sample_dataset(raw_dir, samples_per_class=args.samples_per_class)

    manifest_path = paths_cfg["data"]["cloud_manifest"]
    print(f"Building manifest at: {manifest_path}")
    df = create_cloud_manifest(raw_dir, manifest_path)
    print(f"Manifest created successfully with {len(df)} records.")
    print("Class distribution:")
    print(df["class_name"].value_counts())
    print("\nSplit distribution:")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
