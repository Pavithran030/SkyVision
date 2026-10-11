"""Unified model training pipeline for Vision and Rainfall Nowcasting."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import torchvision.transforms as T

import config
from src.dataset import (
    prepare_sample_dataset,
    create_cloud_manifest,
    create_paired_rainfall_manifest,
    CloudImageDataset,
    get_cloud_transforms,
)
from src.models import (
    CloudClassifier,
    RainfallBaselineModel,
    RainfallEmbeddingModel,
    MultimodalRainfallModel,
    save_cloud_checkpoint,
    save_artifact,
)
from src.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    plot_observed_vs_predicted,
)


def train_vision_model(epochs: int = 5, lr: float = 0.001, batch_size: int = 16):
    """Fine-tune MobileNetV3-Small on sky cloud patterns."""
    print("\n--- Training Task A: Cloud Classifier (MobileNetV3) ---")
    df = pd.read_csv(config.CLOUD_MANIFEST)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    classes = config.DEFAULT_CLASSES
    c2idx = {c: i for i, c in enumerate(classes)}

    train_tf = get_cloud_transforms(is_train=True)
    val_tf = get_cloud_transforms(is_train=False)

    train_ds = CloudImageDataset(train_df, transform=train_tf, class_to_idx=c2idx)
    val_ds = CloudImageDataset(val_df, transform=val_tf, class_to_idx=c2idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = "cpu"
    model = CloudClassifier(num_classes=len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, targets, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                outputs = model(imgs.to(device))
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        is_best = f1 > best_f1
        if is_best:
            best_f1 = f1
            save_cloud_checkpoint(model, config.CLOUD_CHECKPOINT, c2idx)

        print(f"Epoch {epoch:2d}/{epochs:2d} | Val Acc: {acc*100:.1f}% | Val Macro-F1: {f1:.4f} {'[*BEST*]' if is_best else ''}")

    print(f"Vision Training Complete. Checkpoint saved to: {config.CLOUD_CHECKPOINT}")


def train_rainfall_models():
    """Extract visual embeddings and train nowcasting estimators."""
    print("\n--- Training Task B: Rainfall Nowcasting Estimators ---")
    df = pd.read_csv(config.RAINFALL_MANIFEST)

    # Load trained vision model for embeddings
    ckpt_data = torch.load(config.CLOUD_CHECKPOINT, map_location="cpu", weights_only=False)
    vision_model = CloudClassifier(num_classes=ckpt_data["num_classes"], pretrained=False)
    vision_model.load_state_dict(ckpt_data["state_dict"])
    vision_model.eval()

    tf = get_cloud_transforms(is_train=False)
    embeddings = []
    print(f"Extracting CNN embeddings for {len(df)} images...")
    with torch.no_grad():
        for _, row in df.iterrows():
            img_p = Path(row["image_path"])
            if not img_p.is_absolute():
                img_p = config.PROJECT_ROOT / img_p
            img = Image.open(img_p).convert("RGB")
            emb = vision_model.extract_features(tf(img).unsqueeze(0)).numpy()[0]
            embeddings.append(emb)

    X_emb = np.array(embeddings)
    X_weather = df[config.MULTIMODAL_FEATURES].to_numpy()
    X_multi = np.hstack([X_emb, X_weather])

    y_cat = df["rain_category"].to_numpy()
    y_mm = df["rain_mm_next_hour"].to_numpy()

    train_m = (df["split"] == "train").to_numpy()
    test_m = (df["split"] == "test").to_numpy()

    # 1. Baseline
    baseline = RainfallBaselineModel().fit(X_emb[train_m], y_cat[train_m], y_mm[train_m])
    base_cat_p = baseline.predict_category(X_emb[test_m])
    base_mm_p = baseline.predict_amount(X_emb[test_m])

    # 2. Image Random Forest
    rf_img = RainfallEmbeddingModel().fit(X_emb[train_m], y_cat[train_m], y_mm[train_m])
    img_cat_p = rf_img.predict_category(X_emb[test_m])
    img_mm_p = rf_img.predict_amount(X_emb[test_m])

    # 3. Multimodal
    rf_multi = MultimodalRainfallModel().fit(X_multi[train_m], y_cat[train_m])
    multi_cat_p = rf_multi.predict(X_multi[test_m])

    # Evaluate
    y_test_cat = y_cat[test_m]
    y_test_mm = y_mm[test_m]

    base_cls = compute_classification_metrics(y_test_cat, base_cat_p)
    img_cls = compute_classification_metrics(y_test_cat, img_cat_p)
    multi_cls = compute_classification_metrics(y_test_cat, multi_cat_p)

    base_reg = compute_regression_metrics(y_test_mm, base_mm_p)
    img_reg = compute_regression_metrics(y_test_mm, img_mm_p)

    print("\n--- Test Set Performance ---")
    print(f"Majority Baseline: Macro-F1 = {base_cls['macro_f1']:.4f} | MAE = {base_reg['mae_mm']:.2f} mm")
    print(f"Image Embeddings:  Macro-F1 = {img_cls['macro_f1']:.4f} | MAE = {img_reg['mae_mm']:.2f} mm")
    print(f"Multimodal Model:  Macro-F1 = {multi_cls['macro_f1']:.4f}")

    # Save artifacts
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "label_source": "open_meteo_proxy_reanalysis",
        "test_macro_f1_image": img_cls["macro_f1"],
        "test_mae_mm_image": img_reg["mae_mm"],
        "test_macro_f1_multimodal": multi_cls["macro_f1"],
    }
    save_artifact(rf_img.classifier, config.RAINFALL_CLASSIFIER, metadata)
    save_artifact(rf_img.regressor, config.RAINFALL_REGRESSOR, metadata)
    save_artifact(rf_multi, config.MULTIMODAL_CLASSIFIER, metadata)
    with open(config.METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    plot_observed_vs_predicted(y_test_mm, img_mm_p, config.REPORTS_DIR / "rainfall_scatter.png")
    print(f"Artifacts saved to {config.ARTIFACTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train SkyVision models.")
    parser.add_argument("--prepare", action="store_true", help="Generate sample dataset & manifests.")
    parser.add_argument("--vision", action="store_true", help="Train cloud vision classifier.")
    parser.add_argument("--rainfall", action="store_true", help="Train rainfall estimators.")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline.")
    args = parser.parse_args()

    if args.prepare or args.all:
        print("Preparing sample dataset and manifests...")
        prepare_sample_dataset(config.SAMPLES_DIR, samples_per_class=24)
        create_cloud_manifest(config.SAMPLES_DIR, config.CLOUD_MANIFEST)
        create_paired_rainfall_manifest(config.CLOUD_MANIFEST, config.RAINFALL_MANIFEST)

    if args.vision or args.all:
        train_vision_model(epochs=5)

    if args.rainfall or args.all:
        train_rainfall_models()


if __name__ == "__main__":
    main()
