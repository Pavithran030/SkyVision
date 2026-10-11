"""Evaluation metrics and publication-quality figure generation."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
)


def compute_classification_metrics(
    y_true: List[str] | np.ndarray,
    y_pred: List[str] | np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute accuracy, balanced accuracy, macro-F1, per-class metrics, and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        lbl: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
        for i, lbl in enumerate(labels)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute MAE and RMSE in millimetres with non-negative post-processing."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    return {
        "mae_mm": float(mean_absolute_error(y_t, y_p)),
        "rmse_mm": float(root_mean_squared_error(y_t, y_p)),
    }


def plot_confusion_matrix(cm: List[List[int]], labels: List[str], save_path: Path) -> None:
    """Generate and save confusion matrix heatmap."""
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title="Cloud Pattern Confusion Matrix (Test Split)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(cm_arr[i, j], "d"), ha="center", va="center", color="white" if cm_arr[i, j] > thresh else "black")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_observed_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Generate and save scatter plot comparing observed and predicted rainfall (mm)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#2ca02c", s=35)
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)), 5.0)
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.5, label="Ideal 1:1")
    ax.set_xlabel("Observed Rainfall (mm)")
    ax.set_ylabel("Predicted Rainfall (mm)")
    ax.set_title("Rainfall Nowcasting: Observed vs. Predicted")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
