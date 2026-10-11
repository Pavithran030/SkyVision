"""Script to train and evaluate rainfall nowcasting models."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from cloud_rainfall.config import PROJECT_ROOT, get_paths_config, get_rainfall_model_config
from cloud_rainfall.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from cloud_rainfall.evaluation.plots import plot_observed_vs_predicted
from cloud_rainfall.models.cloud_classifier import load_cloud_checkpoint
from cloud_rainfall.models.rainfall_estimator import (
    RainfallBaselineModel,
    RainfallEmbeddingModel,
    MultimodalRainfallModel,
    save_rainfall_artifact,
)


def extract_manifest_embeddings(manifest_df: pd.DataFrame, checkpoint_path: Path) -> np.ndarray:
    """Extract latent vision feature embeddings for all manifest images."""
    model, ckpt = load_cloud_checkpoint(checkpoint_path, device="cpu")
    model.eval()

    img_size = ckpt.get("image_size", 224)
    mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("normalize_std", [0.229, 0.224, 0.225])

    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    embeddings = []
    print(f"Extracting CNN embeddings for {len(manifest_df)} paired images...")
    with torch.no_grad():
        for _, row in manifest_df.iterrows():
            img_p = Path(row["image_path"])
            if not img_p.is_absolute():
                img_p = PROJECT_ROOT / img_p
            img = Image.open(img_p).convert("RGB")
            t = tf(img).unsqueeze(0)
            emb = model.extract_features(t).cpu().numpy()[0]
            embeddings.append(emb)

    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Train rainfall estimation models.")
    parser.add_argument("--rainfall-manifest", type=str, default=None, help="Path to paired rainfall manifest.")
    parser.add_argument("--cloud-checkpoint", type=str, default=None, help="Path to cloud classifier checkpoint.")
    args = parser.parse_args()

    paths_cfg = get_paths_config()
    rain_cfg = get_rainfall_model_config()

    rain_manifest_p = Path(args.rainfall_manifest or paths_cfg["data"]["rainfall_manifest"])
    if not rain_manifest_p.is_absolute():
        rain_manifest_p = PROJECT_ROOT / rain_manifest_p

    cloud_ckpt_p = Path(args.cloud_checkpoint or paths_cfg["artifacts"]["cloud_checkpoint"])
    if not cloud_ckpt_p.is_absolute():
        cloud_ckpt_p = PROJECT_ROOT / cloud_ckpt_p

    if not rain_manifest_p.exists():
        raise FileNotFoundError(f"Rainfall manifest not found: {rain_manifest_p}. Run scripts/build_rainfall_pairs.py first.")
    if not cloud_ckpt_p.exists():
        raise FileNotFoundError(f"Cloud checkpoint not found: {cloud_ckpt_p}. Run scripts/train_cloud_classifier.py first.")

    df = pd.read_csv(rain_manifest_p)
    valid_df = df[df["rain_category"].notna() & df["rain_mm_next_hour"].notna()].copy()
    if len(valid_df) < 5:
        raise ValueError(f"Too few valid paired samples ({len(valid_df)}) to train rainfall models.")

    print(f"\n=======================================================")
    print(f" Training Rainfall Nowcasting Models (Task B)")
    print(f"=======================================================")
    print(f"Total valid paired samples: {len(valid_df)}")
    print(f"Label Source:              {valid_df['rain_source'].iloc[0]}")
    print(f"Horizon:                   Next 1 Hour ([t, t+1h))")

    # Extract CNN embeddings
    X_embeddings = extract_manifest_embeddings(valid_df, cloud_ckpt_p)

    # Meteorological features
    met_features = rain_cfg["features"]["multimodal"]
    X_weather = valid_df[met_features].to_numpy()
    X_multimodal = np.hstack([X_embeddings, X_weather])

    y_category = valid_df["rain_category"].to_numpy()
    y_mm = valid_df["rain_mm_next_hour"].to_numpy()

    # Split masks
    train_mask = (valid_df["split"] == "train").to_numpy()
    val_mask = (valid_df["split"] == "val").to_numpy()
    test_mask = (valid_df["split"] == "test").to_numpy()

    # If test mask is empty, fallback to 20% random test
    if not np.any(test_mask):
        train_mask[: int(len(valid_df) * 0.8)] = True
        test_mask[int(len(valid_df) * 0.8) :] = True

    # 1. Train Baseline Model
    baseline = RainfallBaselineModel()
    baseline.fit(X_embeddings[train_mask], y_category[train_mask], y_mm[train_mask])
    base_cat_preds = baseline.predict_category(X_embeddings[test_mask])
    base_mm_preds = baseline.predict_amount(X_embeddings[test_mask])

    # 2. Train Image-only Random Forest
    rf_image_model = RainfallEmbeddingModel(random_state=42)
    rf_image_model.fit(X_embeddings[train_mask], y_category[train_mask], y_mm[train_mask])
    img_cat_preds = rf_image_model.predict_category(X_embeddings[test_mask])
    img_mm_preds = rf_image_model.predict_amount(X_embeddings[test_mask])

    # 3. Train Multimodal Model
    mm_model = MultimodalRainfallModel(feature_names=met_features, random_state=42)
    mm_model.fit(X_multimodal[train_mask], y_category[train_mask])
    mm_cat_preds = mm_model.predict(X_multimodal[test_mask])

    # Evaluate on held-out test split
    y_test_cat = y_category[test_mask]
    y_test_mm = y_mm[test_mask]

    base_cls_metrics = compute_classification_metrics(y_test_cat, base_cat_preds)
    img_cls_metrics = compute_classification_metrics(y_test_cat, img_cat_preds)
    mm_cls_metrics = compute_classification_metrics(y_test_cat, mm_cat_preds)

    base_reg_metrics = compute_regression_metrics(y_test_mm, base_mm_preds)
    img_reg_metrics = compute_regression_metrics(y_test_mm, img_mm_preds)

    print("\n--- Model Comparison on Held-Out Test Set ---")
    print(f"{'Model Variant':<26} {'Macro-F1':<12} {'Accuracy':<12} {'MAE (mm)':<12} {'RMSE (mm)':<12}")
    print("-" * 74)
    print(f"{'Majority/Median Baseline':<26} {base_cls_metrics['macro_f1']:<12.4f} {base_cls_metrics['accuracy']*100:<12.1f}% {base_reg_metrics['mae_mm_all']:<12.2f} {base_reg_metrics['rmse_mm_all']:<12.2f}")
    print(f"{'Image-Only CNN Embeddings':<26} {img_cls_metrics['macro_f1']:<12.4f} {img_cls_metrics['accuracy']*100:<12.1f}% {img_reg_metrics['mae_mm_all']:<12.2f} {img_reg_metrics['rmse_mm_all']:<12.2f}")
    print(f"{'Multimodal (Image + Weather)':<26} {mm_cls_metrics['macro_f1']:<12.4f} {mm_cls_metrics['accuracy']*100:<12.1f}% {'N/A':<12} {'N/A':<12}")

    # Save artifacts
    artifacts_dir = PROJECT_ROOT / paths_cfg["artifacts"]["dir"]
    classifier_path = artifacts_dir / "rainfall_classifier_best.joblib"
    regressor_path = artifacts_dir / "rainfall_regressor_best.joblib"
    multimodal_path = artifacts_dir / "multimodal_classifier_best.joblib"

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "horizon": "next_1_hour",
        "label_source": valid_df["rain_source"].iloc[0],
        "test_macro_f1_image": img_cls_metrics["macro_f1"],
        "test_mae_mm_image": img_reg_metrics["mae_mm_all"],
        "test_macro_f1_multimodal": mm_cls_metrics["macro_f1"],
        "features_multimodal": met_features,
    }

    save_rainfall_artifact(rf_image_model.classifier, classifier_path, metadata)
    save_rainfall_artifact(rf_image_model.regressor, regressor_path, metadata)
    save_rainfall_artifact(mm_model, multimodal_path, metadata)

    with open(artifacts_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save scatter plot of observed vs predicted
    scatter_path = PROJECT_ROOT / paths_cfg["reports"]["figures_dir"] / "rainfall_observed_vs_predicted.png"
    plot_observed_vs_predicted(y_test_mm, img_mm_preds, save_path=scatter_path)

    print(f"\nArtifacts successfully saved to: {artifacts_dir}")
    print(f"Observation vs Prediction plot:  {scatter_path}")


if __name__ == "__main__":
    main()
