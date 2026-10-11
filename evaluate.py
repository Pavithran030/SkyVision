"""Evaluate trained SkyVision models on held-out test splits."""

import pandas as pd
from PIL import Image
import torch

import config
from src.dataset import validate_image_file
from src.inference import CloudInferenceEngine
from src.metrics import compute_classification_metrics, plot_confusion_matrix


def main():
    print("\n=======================================================")
    print(" Evaluating SkyVision Cloud Classifier on Test Split")
    print("=======================================================")

    if not config.CLOUD_CHECKPOINT.exists():
        print(f"Error: Checkpoint {config.CLOUD_CHECKPOINT} not found. Run `python train.py --vision` first.")
        return

    engine = CloudInferenceEngine(config.CLOUD_CHECKPOINT)
    df = pd.read_csv(config.CLOUD_MANIFEST)
    test_df = df[df["split"] == "test"].copy()

    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        img_p = config.PROJECT_ROOT / row["image_path"]
        img = Image.open(img_p).convert("RGB")
        res = engine.predict(img)
        y_true.append(row["class_name"])
        y_pred.append(res["predicted_class"])

    metrics = compute_classification_metrics(y_true, y_pred)

    print(f"Held-Out Test Samples: {len(test_df)}")
    print(f"Overall Accuracy:      {metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy:     {metrics['balanced_accuracy']*100:.2f}%")
    print(f"Macro-F1 Score:        {metrics['macro_f1']:.4f}")

    print("\nPer-Class Breakdown:")
    for c, v in metrics["per_class"].items():
        print(f"  - {c:<22}: Precision: {v['precision']*100:5.1f}% | Recall: {v['recall']*100:5.1f}% | F1: {v['f1']:.4f}")

    cm_path = config.REPORTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
