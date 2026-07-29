"""Confusion matrix visualization for multi-class classification."""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.visualization.display import show_saved_figure


def _confusion_figsize(n_classes, base=0.28, min_size=10, max_size=40):
    size = min(max(min_size, n_classes * base), max_size)
    return (size, size)


def plot_confusion_matrix(
    predictions,
    labels,
    class_names,
    normalize=True,
    figsize=None,
    show=True,
    save_path="outputs/confusion_matrix.png",
    annot_threshold=20,
):
    """
    Plot confusion matrix from pre-computed predictions.

    For many classes, cell annotations are disabled and the figure is scaled up
    so labels stay readable when zooming the saved image.
    """
    n_classes = len(class_names)
    if figsize is None:
        figsize = _confusion_figsize(n_classes)

    cm = confusion_matrix(labels, predictions, labels=list(range(n_classes)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1

    if normalize:
        cm_display = cm.astype("float") / row_sums
        title = f"Normalized Confusion Matrix ({n_classes} classes)"
        fmt = ".2f"
    else:
        cm_display, fmt, title = cm, "d", f"Confusion Matrix ({n_classes} classes)"

    annotate = n_classes <= annot_threshold
    tick_fontsize = 6 if n_classes > 40 else 8 if n_classes > 20 else 10

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=annotate,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.6},
        linewidths=0.0,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    plt.xticks(rotation=90, ha="center")
    plt.yticks(rotation=0)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    plt.close(fig)

    return fig, cm


def plot_top_confused_pairs(
    predictions,
    labels,
    class_names,
    top_k=25,
    figsize=(12, 8),
    show=True,
    save_path="outputs/top_confused_pairs.png",
):
    """Plot the most common off-diagonal confusions."""
    n_classes = len(class_names)
    cm = confusion_matrix(labels, predictions, labels=list(range(n_classes)))

    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append((count, class_names[i], class_names[j]))

    pairs.sort(reverse=True)
    pairs = pairs[:top_k]
    if not pairs:
        print("No confused pairs to plot.")
        return None

    counts = [p[0] for p in pairs]
    pair_labels = [f"{true_name} → {pred_name}" for _, true_name, pred_name in pairs]

    fig_height = max(6, 0.35 * len(pairs))
    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    y = np.arange(len(pairs))
    ax.barh(y, counts, color="steelblue", edgecolor="darkblue", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Top {len(pairs)} Confused Class Pairs")
    ax.grid(axis="x", alpha=0.3)

    for yi, count in zip(y, counts):
        ax.text(count + 0.1, yi, str(count), va="center", fontsize=8)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    plt.close(fig)
    return fig
