"""Plotting utilities for evaluation and visualizations."""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: List[List[int]] | np.ndarray,
    labels: List[str],
    save_path: Optional[Path | str] = None,
    title: str = "Cloud Classification Confusion Matrix",
) -> plt.Figure:
    """Generate and optionally save confusion matrix heatmap."""
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                format(cm_arr[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_probabilities_bar(
    probabilities: Dict[str, float],
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Generate horizontal bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    classes = list(reversed(list(probabilities.keys())))
    probs = [probabilities[c] * 100 for c in classes]

    colors = ["#1f77b4" if p < max(probs) else "#ff7f0e" for p in probs]
    bars = ax.barh(classes, probs, color=colors, edgecolor="black", alpha=0.85)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Cloud Pattern Classification Probabilities")

    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_observed_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Scatter plot of observed vs predicted precipitation (mm)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#2ca02c", edgecolors="none", s=30)

    max_val = max(float(np.max(y_true)), float(np.max(y_pred)), 5.0)
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.5, label="Ideal 1:1")

    ax.set_xlabel("Observed Rainfall (mm)")
    ax.set_ylabel("Predicted Rainfall (mm)")
    ax.set_title("Rainfall Estimation: Observed vs. Predicted")
    ax.legend()
    fig.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig
