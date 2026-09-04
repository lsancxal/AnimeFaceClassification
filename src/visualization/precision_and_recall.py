"""Precision / recall plots and metric calculations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

from src.visualization.io import save_and_show_figure
from src.visualization.paths import output_path


def calculate_precision_recall(predictions, labels, class_names) -> dict:
    """Calculate precision, recall, and F1 for all classes."""
    class_labels = list(range(len(class_names)))
    return {
        "precision": precision_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ),
        "f1": f1_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ),
        "macro_precision": precision_score(
            labels, predictions, labels=class_labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            labels, predictions, labels=class_labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            labels, predictions, labels=class_labels, average="macro", zero_division=0
        ),
    }


def plot_precision_recall_combined(
    predictions,
    labels,
    class_names,
    figsize=None,
    show: bool = True,
    save_path: str | None = None,
    sort_by: str = "f1",
):
    """
    Plot precision and recall as a tall horizontal bar chart.
    Classes are sorted by F1 by default so weak classes are easy to find.
    """
    if save_path is None:
        save_path = output_path("precision_recall_combined.png")

    metrics = calculate_precision_recall(predictions, labels, class_names)
    precision, recall, f1 = metrics["precision"], metrics["recall"], metrics["f1"]

    if sort_by == "f1":
        order = np.argsort(f1)
    elif sort_by == "name":
        order = np.argsort(class_names)
    else:
        order = np.arange(len(class_names))

    names = [class_names[i] for i in order]
    precision = precision[order]
    recall = recall[order]

    n_classes = len(names)
    if figsize is None:
        figsize = (12, max(8, 0.28 * n_classes))

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n_classes)
    height = 0.35

    ax.barh(
        y - height / 2,
        precision,
        height,
        label="Precision",
        color="forestgreen",
        edgecolor="darkgreen",
        alpha=0.85,
    )
    ax.barh(
        y + height / 2,
        recall,
        height,
        label="Recall",
        color="steelblue",
        edgecolor="darkblue",
        alpha=0.85,
    )
    ax.axvline(
        metrics["macro_f1"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Macro F1: {metrics['macro_f1']:.2%}",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.05)
    ax.set_title(
        f"Precision and Recall per Class (sorted by F1, {n_classes} classes)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_and_show_figure(fig, save_path, show=show)
    return fig, metrics["precision"], metrics["recall"]
