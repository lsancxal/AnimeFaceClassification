
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, classification_report


def calculate_precision_recall(predictions, labels, class_names):
    """
    Calculate precision and recall for all classes.

    Args:
        predictions: Model predictions as numpy array.
        labels: True labels as numpy array.

    Returns:
        dict: Contains precision_per_class, recall_per_class, macro_precision, macro_recall
    """
    class_labels = list(range(len(class_names)))
    return {
        'precision': precision_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ),
        'recall': recall_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ),
        'macro_precision': precision_score(
            labels, predictions, labels=class_labels, average='macro', zero_division=0
        ),
        'macro_recall': recall_score(
            labels, predictions, labels=class_labels, average='macro', zero_division=0
        ),
    }


def _plot_metric_bar(class_names, values, macro_avg, metric_name, color_map, edge_color, figsize, show, save_path=None):
    """
    Internal helper to plot a metric bar chart.
    
    Args:
        values: Per-class metric values.
        macro_avg: Macro average value.
        metric_name: Name of the metric ('Precision' or 'Recall').
        color_map: Matplotlib colormap to use.
        edge_color: Edge color for bars.
        figsize: Figure size.
        show: Whether to call plt.show().
        
    Returns:
        fig: The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(class_names))
    colors = color_map(0.3 + 0.7 * values)
    bars = ax.bar(x, values, color=colors, edgecolor=edge_color, linewidth=1.2)
    
    # Macro average line
    ax.axhline(y=macro_avg, color='red', linestyle='--', linewidth=2, 
               label=f'Macro Avg: {macro_avg:.2%}')
    
    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    
    return fig


def plot_precision(predictions, labels, figsize=(10, 6), show=True, save_path=None):
    """
    Plot precision for all 10 classes as a bar chart.
    
    Args:
        predictions: Model predictions as numpy array.
        labels: True labels as numpy array.
        figsize: Figure size as (width, height).
        show: Whether to call plt.show().
        
    Returns:
        tuple: (fig, precision_per_class)
    """
    metrics = calculate_precision_recall(predictions, labels)
    fig = _plot_metric_bar(
        metrics['precision'], metrics['macro_precision'],
        'Precision', plt.cm.Greens, 'darkgreen', figsize, show, save_path
    )
    return fig, metrics['precision']


def plot_recall(predictions, labels, figsize=(10, 6), show=True, save_path=None):
    """
    Plot recall for all 10 classes as a bar chart.
    
    Args:
        predictions: Model predictions as numpy array.
        labels: True labels as numpy array.
        figsize: Figure size as (width, height).
        show: Whether to call plt.show().
        
    Returns:
        tuple: (fig, recall_per_class)
    """
    metrics = calculate_precision_recall(predictions, labels)
    fig = _plot_metric_bar(
        metrics['recall'], metrics['macro_recall'],
        'Recall', plt.cm.Blues, 'darkblue', figsize, show, save_path
    )
    return fig, metrics['recall']


def plot_precision_recall_combined(predictions, labels, class_names, figsize=(12, 6), show=True, save_path='outputs/precision_recall_combined.png'):
    """
    Plot precision and recall side by side for all classes.

    Args:
        predictions: Model predictions as numpy array.
        labels: True labels as numpy array.
        figsize: Figure size as (width, height).
        show: Whether to call plt.show().
        save_path: Path to save the figure. Pass None to skip saving.

    Returns:
        tuple: (fig, precision_per_class, recall_per_class)
    """
    metrics = calculate_precision_recall(predictions, labels, class_names)
    precision, recall = metrics['precision'], metrics['recall']
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(class_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, precision, width, label='Precision', 
                   color='forestgreen', edgecolor='darkgreen', alpha=0.8)
    bars2 = ax.bar(x + width/2, recall, width, label='Recall',
                   color='steelblue', edgecolor='darkblue', alpha=0.8)
    
    for bar, val in zip(bars1, precision):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.0%}', ha='center', va='bottom', fontsize=8)
    
    for bar, val in zip(bars2, recall):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.0%}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision and Recall per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    
    # Print summary
    print(f"\nPrecision and Recall Summary:")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10}")
    print("-" * 34)
    for name, prec, rec in zip(class_names, precision, recall):
        print(f"{name:<12} {prec:>10.2%} {rec:>10.2%}")
    print("-" * 34)
    print(f"{'Macro Avg':<12} {metrics['macro_precision']:>10.2%} {metrics['macro_recall']:>10.2%}")
    
    return fig, precision, recall


def print_classification_report(predictions, labels, class_names):
    """
    Print a detailed classification report with precision, recall, and F1-score.
    
    Args:
        predictions: Model predictions as numpy array.
        labels: True labels as numpy array.
    """
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    ))