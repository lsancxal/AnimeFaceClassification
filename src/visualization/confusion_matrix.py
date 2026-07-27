"""Confusion matrix visualization for Fashion-MNIST classification."""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.visualization.display import show_saved_figure



def plot_confusion_matrix(predictions, labels, class_names, normalize=True, figsize=(12, 10), show=True, save_path="outputs/confusion_matrix.png"):
    """
    Plot confusion matrix from pre-computed predictions.
    
    Args:
        predictions: Model predictions as numpy array.
        labels: True labels as numpy array.
        normalize: If True, normalize the confusion matrix by row.
        figsize: Figure size as (width, height).
        show: Whether to call plt.show().
        
    Returns:
        tuple: (fig, confusion_matrix_array)
    """
    cm = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt, title = '.2f', 'Normalized Confusion Matrix'
    else:
        cm_display, fmt, title = cm, 'd', 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, square=True, cbar_kws={'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    plt.close(fig)
    
    return fig, cm


def plot_confusion_matrix_with_stats(predictions, labels, class_names, figsize=(14, 10), show=True, save_path="outputs/confusion_matrix_with_stats.png"):
    """
    Plot confusion matrix with per-class accuracy statistics.
    
    Args:
        predictions: Model predictions as numpy array.
        labels: True labels as numpy array.
        figsize: Figure size as (width, height).
        show: Whether to call plt.show().
        
    Returns:
        tuple: (fig, confusion_matrix_array, per_class_accuracy)
    """
    cm = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                    gridspec_kw={'width_ratios': [3, 1]})
    
    # Confusion matrix heatmap
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax1, square=True, cbar_kws={'shrink': 0.8}
    )
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Per-class accuracy bar chart
    colors = plt.cm.Blues(per_class_accuracy)
    bars = ax2.barh(range(len(class_names)), per_class_accuracy, color=colors)
    ax2.set_yticks(range(len(class_names)))
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Accuracy', fontsize=12)
    ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    
    for bar, acc in zip(bars, per_class_accuracy):
        ax2.text(acc + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2%}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    plt.close(fig)
    
    # Print summary
    overall_accuracy = (predictions == labels).sum() / len(labels)
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
    print(f"\nPer-Class Accuracy:")
    for name, acc in zip(class_names, per_class_accuracy):
        print(f"  {name:12s}: {acc:.2%}")
    
    return fig, cm, per_class_accuracy