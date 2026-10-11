"""Dataset inspection and manifest generation module."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from cloud_rainfall.config import PROJECT_ROOT
from cloud_rainfall.data.validation import validate_image_file

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for duplicate detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def scan_cloud_dataset_folder(folder_path: Path | str) -> pd.DataFrame:
    """
    Scan a directory of sky images structured by class folders.
    Directory structure:
      <folder_path>/
        <class_A>/img1.jpg, img2.png
        <class_B>/img3.jpg, ...
        
    Returns a DataFrame containing details of every scanned file.
    """
    root = Path(folder_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

    records = []
    seen_hashes: Dict[str, str] = {}

    # Look for subdirectories as classes
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        # If no subdirectories, look for images directly in root (unlabelled / single class)
        subdirs = [root]

    for class_dir in subdirs:
        class_name = class_dir.name if class_dir != root else "unclassified"
        for file_path in class_dir.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                rel_path = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                rel_path = str(file_path).replace("\\", "/")

            # Compute hash
            try:
                f_hash = compute_file_hash(file_path)
                is_duplicate = f_hash in seen_hashes
                if not is_duplicate:
                    seen_hashes[f_hash] = str(file_path)
            except Exception:
                f_hash = ""
                is_duplicate = False

            # Validate image
            val_res = validate_image_file(file_path)
            warning_text = "; ".join(val_res.warnings) if val_res.warnings else ""

            records.append({
                "image_path": str(file_path).replace("\\", "/"),
                "relative_path": rel_path,
                "class_name": class_name,
                "width": val_res.width,
                "height": val_res.height,
                "sha256": f_hash,
                "is_duplicate": is_duplicate,
                "is_valid": val_res.is_valid,
                "error_message": val_res.error_message or "",
                "warnings": warning_text,
            })

    df = pd.DataFrame(records)
    return df


def create_cloud_manifest(
    dataset_dir: Path | str,
    output_manifest_path: Path | str,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Scan folder, filter valid samples, assign stratified train/val/test splits,
    and save manifest CSV.
    """
    df = scan_cloud_dataset_folder(dataset_dir)
    if df.empty:
        raise ValueError(f"No supported images found in {dataset_dir}")

    # Assign label IDs
    unique_classes = sorted(df["class_name"].unique())
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    df["label_id"] = df["class_name"].map(class_to_id)

    # Filter valid images for splitting
    valid_df = df[df["is_valid"] & (~df["is_duplicate"])].copy()

    # If dataset has very few images per class, handle gracefully
    df["split"] = "unassigned"

    if len(valid_df) > 0:
        # Check if stratify is possible
        class_counts = valid_df["class_name"].value_counts()
        can_stratify = (class_counts.min() >= 3) and (len(unique_classes) > 1)

        total_holdout = val_split + test_split
        if total_holdout > 0 and len(valid_df) >= 3:
            train_idx, holdout_idx = train_test_split(
                valid_df.index,
                test_size=total_holdout,
                random_state=random_seed,
                stratify=valid_df["class_name"] if can_stratify else None,
            )
            val_proportion = val_split / total_holdout
            val_idx, test_idx = train_test_split(
                holdout_idx,
                test_size=(1.0 - val_proportion),
                random_state=random_seed,
                stratify=valid_df.loc[holdout_idx, "class_name"] if can_stratify else None,
            )
            df.loc[train_idx, "split"] = "train"
            df.loc[val_idx, "split"] = "val"
            df.loc[test_idx, "split"] = "test"
        else:
            df.loc[valid_df.index, "split"] = "train"

    out_path = Path(output_manifest_path)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_and_validate_manifest(manifest_path: Path | str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a manifest CSV and generate a data quality summary report.
    """
    path = Path(manifest_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_csv(path)
    summary = {
        "total_images": len(df),
        "valid_images": int(df["is_valid"].sum()),
        "invalid_images": int((~df["is_valid"]).sum()),
        "duplicate_images": int(df["is_duplicate"].sum()) if "is_duplicate" in df else 0,
        "classes": df["class_name"].value_counts().to_dict(),
        "splits": df["split"].value_counts().to_dict() if "split" in df else {},
    }
    return df, summary
