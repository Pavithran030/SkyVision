"""Model evaluation script for held-out test splits."""

import argparse
from pathlib import Path
import pandas as pd
import torch

from cloud_rainfall.config import PROJECT_ROOT, get_paths_config
from cloud_rainfall.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from cloud_rainfall.evaluation.plots import (
    plot_confusion_matrix,
    plot_observed_vs_predicted,
)
from cloud_rainfall.models.inference import CloudInferenceEngine


def evaluate_cloud_classifier(manifest_path: Path, checkpoint_path: Path, out_dir: Path):
    print("\n=======================================================")
    print(" Evaluating Cloud Pattern Classifier on Held-Out Test Split")
    print("=======================================================")

    engine = CloudInferenceEngine(checkpoint_path)
    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].copy()

    if test_df.empty:
        print("Warning: No test samples found in manifest!")
        return

    print(f"Total held-out test samples: {len(test_df)}")

    y_true = []
    y_pred = []
    confidences = []

    for _, row in test_df.iterrows():
        img_p = Path(row["image_path"])
        if not img_p.is_absolute():
            img_p = PROJECT_ROOT / img_p
        res = engine.predict(img_p)
        if res["is_valid"]:
            y_true.append(row["class_name"])
            y_pred.append(res["predicted_class"])
            confidences.append(res["confidence"])

    labels = sorted(list(set(y_true) | set(y_pred)))
    metrics = compute_classification_metrics(y_true, y_pred, labels=labels)

    test_df["predicted_class"] = y_pred
    test_df["confidence"] = confidences
    test_df["correct"] = [t == p for t, p in zip(y_true, y_pred)]

    out_csv = out_dir / "cloud_test_predictions.csv"
    test_df.to_csv(out_csv, index=False)
    print(f"Per-sample predictions saved to: {out_csv}")

    # Plot confusion matrix
    cm_fig_path = out_dir / "figures" / "cloud_confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], labels=labels, save_path=cm_fig_path)
    print(f"Confusion matrix plot saved to:   {cm_fig_path}")

    print("\n--- Summary Performance Metrics ---")
    print(f"Overall Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"Macro-F1 Score:    {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1 Score: {metrics['weighted_f1']:.4f}")

    print("\n--- Per-Class Performance ---")
    print(f"{'Class':<22} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 62)
    for cls_name, vals in metrics["per_class"].items():
        print(
            f"{cls_name:<22} "
            f"{vals['precision']*100:<10.1f} "
            f"{vals['recall']*100:<10.1f} "
            f"{vals['f1']:<10.4f} "
            f"{vals['support']:<8}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on held-out test splits.")
    parser.add_argument("--cloud-manifest", type=str, default=None, help="Cloud manifest CSV.")
    parser.add_argument("--cloud-checkpoint", type=str, default=None, help="Cloud model checkpoint.")
    args = parser.parse_args()

    paths_cfg = get_paths_config()
    manifest_p = Path(args.cloud_manifest or paths_cfg["data"]["cloud_manifest"])
    if not manifest_p.is_absolute():
        manifest_p = PROJECT_ROOT / manifest_p

    ckpt_p = Path(args.cloud_checkpoint or paths_cfg["artifacts"]["cloud_checkpoint"])
    if not ckpt_p.is_absolute():
        ckpt_p = PROJECT_ROOT / ckpt_p

    out_dir = PROJECT_ROOT / paths_cfg["reports"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if ckpt_p.exists() and manifest_p.exists():
        evaluate_cloud_classifier(manifest_p, ckpt_p, out_dir)
    else:
        print(f"Cannot evaluate: manifest ({manifest_p}) or checkpoint ({ckpt_p}) not found.")


if __name__ == "__main__":
    main()
