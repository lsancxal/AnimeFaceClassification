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
        cm_display = cm.astype('float') / row_sums
        title = f'Normalized Confusion Matrix ({n_classes} classes)'
        fmt = '.2f'
    else:
        cm_display, fmt, title = cm, 'd', f'Confusion Matrix ({n_classes} classes)'

    annotate = n_classes <= annot_threshold
    tick_fontsize = 6 if n_classes > 40 else 8 if n_classes > 20 else 10

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=annotate,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={'shrink': 0.6},
        linewidths=0.0,
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
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
    save_path='outputs/top_confused_pairs.png',
):
    """
    Plot the most common off-diagonal confusions.
    Much more readable than a full 130x130 matrix.
    """
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
    ax.barh(y, counts, color='steelblue', edgecolor='darkblue', alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title(f'Top {len(pairs)} Confused Class Pairs')
    ax.grid(axis='x', alpha=0.3)

    for yi, count in zip(y, counts):
        ax.text(count + 0.1, yi, str(count), va='center', fontsize=8)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    plt.close(fig)
    return fig


def plot_confusion_matrix_with_stats(
    predictions,
    labels,
    class_names,
    figsize=(14, 10),
    show=True,
    save_path="outputs/confusion_matrix_with_stats.png",
):
    """
    Plot confusion matrix with per-class accuracy statistics.
    Prefer plot_top_confused_pairs + HTML metrics for many classes.
    """
    cm = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    row_sums = cm.sum(axis=1)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    per_class_accuracy = cm.diagonal() / row_sums_safe
    cm_normalized = cm.astype('float') / row_sums_safe[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1]}
    )

    annotate = len(class_names) <= 20
    sns.heatmap(
        cm_normalized,
        annot=annotate,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
        square=True,
        cbar_kws={'shrink': 0.8},
    )
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    colors = plt.cm.Blues(per_class_accuracy)
    bars = ax2.barh(range(len(class_names)), per_class_accuracy, color=colors)
    ax2.set_yticks(range(len(class_names)))
    ax2.set_yticklabels(class_names, fontsize=7)
    ax2.set_xlabel('Accuracy', fontsize=12)
    ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()

    for bar, acc in zip(bars, per_class_accuracy):
        ax2.text(
            acc + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f'{acc:.2%}',
            va='center',
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    plt.close(fig)

    overall_accuracy = (predictions == labels).sum() / len(labels)
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
    print(f"\nPer-Class Accuracy:")
    for name, acc in zip(class_names, per_class_accuracy):
        print(f"  {name:12s}: {acc:.2%}")

    return fig, cm, per_class_accuracy
