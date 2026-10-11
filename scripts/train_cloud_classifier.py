"""Script to train cloud pattern classifier using transfer learning."""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from cloud_rainfall.config import PROJECT_ROOT, get_paths_config, get_cloud_model_config
from cloud_rainfall.data.datasets import CloudImageDataset, get_cloud_transforms
from cloud_rainfall.models.cloud_classifier import (
    build_cloud_classifier,
    save_cloud_checkpoint,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * len(targets)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    val_loss = total_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return val_loss, val_acc, val_macro_f1


def main():
    parser = argparse.ArgumentParser(description="Train cloud pattern classifier.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze pretrained feature backbone.")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on subset.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to cloud manifest CSV.")
    parser.add_argument("--checkpoint-out", type=str, default=None, help="Output checkpoint path.")
    args = parser.parse_args()

    paths_cfg = get_paths_config()
    cloud_cfg = get_cloud_model_config()

    manifest_path = Path(args.manifest or paths_cfg["data"]["cloud_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run scripts/prepare_cloud_dataset.py first."
        )

    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Train or validation split is empty in manifest.")

    if args.smoke_test:
        print("[Smoke-test mode: restricting to subset of samples]")
        train_df = train_df.groupby("class_name").head(4)
        val_df = val_df.groupby("class_name").head(2)

    unique_classes = sorted(df["class_name"].unique())
    class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
    num_classes = len(unique_classes)

    print(f"Dataset summary:")
    print(f"  Classes ({num_classes}): {unique_classes}")
    print(f"  Training samples:   {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")

    # Transforms and Datasets
    img_size = cloud_cfg["model"]["image_size"]
    mean = cloud_cfg["model"]["normalize_mean"]
    std = cloud_cfg["model"]["normalize_std"]

    train_tf = get_cloud_transforms(image_size=img_size, is_train=True, normalize_mean=mean, normalize_std=std)
    val_tf = get_cloud_transforms(image_size=img_size, is_train=False, normalize_mean=mean, normalize_std=std)

    train_ds = CloudImageDataset(train_df, transform=train_tf, class_to_idx=class_to_idx)
    val_ds = CloudImageDataset(val_df, transform=val_tf, class_to_idx=class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = "cpu"
    backbone = cloud_cfg["model"]["backbone"]
    print(f"\nBuilding model ({backbone}) on {device}...")
    model = build_cloud_classifier(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=cloud_cfg["model"]["pretrained"],
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=cloud_cfg["training"]["weight_decay"],
    )

    best_val_f1 = -1.0
    checkpoint_out = Path(args.checkpoint_out or paths_cfg["artifacts"]["cloud_checkpoint"])
    if not checkpoint_out.is_absolute():
        checkpoint_out = PROJECT_ROOT / checkpoint_out

    print(f"\nStarting training for {args.epochs} epochs:")
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_f1 = evaluate(model, val_loader, criterion, device)

        is_best = v_f1 > best_val_f1
        if is_best:
            best_val_f1 = v_f1
            save_cloud_checkpoint(
                model=model,
                save_path=checkpoint_out,
                class_to_idx=class_to_idx,
                image_size=img_size,
                normalize_mean=mean,
                normalize_std=std,
                extra_metadata={
                    "best_epoch": epoch,
                    "val_loss": v_loss,
                    "val_acc": v_acc,
                    "val_macro_f1": v_f1,
                    "trained_at": datetime.now().isoformat(),
                },
            )

        marker = " [*BEST SAVED*]" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train Loss: {t_loss:.4f} Acc: {t_acc*100:.1f}% | "
            f"Val Loss: {v_loss:.4f} Acc: {v_acc*100:.1f}% Macro-F1: {v_f1:.4f}{marker}"
        )

    print(f"\nTraining completed! Best Validation Macro-F1: {best_val_f1:.4f}")
    print(f"Best checkpoint saved to: {checkpoint_out}")


if __name__ == "__main__":
    main()
