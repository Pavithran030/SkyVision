"""Evaluation metrics calculation for vision and rainfall models."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


def compute_classification_metrics(
    y_true: List[str] | np.ndarray,
    y_pred: List[str] | np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive multi-class classification metrics:
    - Accuracy & Balanced Accuracy
    - Macro-F1 & Weighted-F1
    - Per-class precision, recall, f1, and support
    - Confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm,
        "total_samples": len(y_true),
    }


def compute_regression_metrics(
    y_true: np.ndarray | List[float],
    y_pred: np.ndarray | List[float],
    rain_threshold_mm: float = 0.1,
) -> Dict[str, Any]:
    """
    Compute rainfall regression metrics (in mm).
    Evaluates both overall intervals and rainy intervals separately.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    # Documented post-processing: clip negative values to 0
    y_p_clipped = np.clip(y_p, 0.0, None)

    mae_all = float(mean_absolute_error(y_t, y_p_clipped))
    rmse_all = float(root_mean_squared_error(y_t, y_p_clipped))
    r2_all = float(r2_score(y_t, y_p_clipped)) if len(y_t) > 1 and np.var(y_t) > 1e-6 else 0.0

    # Separate rainy intervals
    rainy_mask = y_t >= rain_threshold_mm
    num_rainy = int(np.sum(rainy_mask))

    if num_rainy > 0:
        mae_rainy = float(mean_absolute_error(y_t[rainy_mask], y_p_clipped[rainy_mask]))
        rmse_rainy = float(root_mean_squared_error(y_t[rainy_mask], y_p_clipped[rainy_mask]))
    else:
        mae_rainy = 0.0
        rmse_rainy = 0.0

    return {
        "mae_mm_all": mae_all,
        "rmse_mm_all": rmse_all,
        "r2_all": r2_all,
        "total_samples": len(y_t),
        "rainy_samples": num_rainy,
        "dry_samples": len(y_t) - num_rainy,
        "dry_ratio": float(1.0 - (num_rainy / len(y_t))) if len(y_t) > 0 else 0.0,
        "mae_mm_rainy": mae_rainy,
        "rmse_mm_rainy": rmse_rainy,
    }
