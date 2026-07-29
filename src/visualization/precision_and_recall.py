import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, classification_report, f1_score

from src.visualization.display import show_saved_figure


def calculate_precision_recall(predictions, labels, class_names):
    """
    Calculate precision, recall, and F1 for all classes.
    """
    class_labels = list(range(len(class_names)))
    return {
        'precision': precision_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ),
        'recall': recall_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ),
        'f1': f1_score(
            labels, predictions, labels=class_labels, average=None, zero_division=0
        ),
        'macro_precision': precision_score(
            labels, predictions, labels=class_labels, average='macro', zero_division=0
        ),
        'macro_recall': recall_score(
            labels, predictions, labels=class_labels, average='macro', zero_division=0
        ),
        'macro_f1': f1_score(
            labels, predictions, labels=class_labels, average='macro', zero_division=0
        ),
    }


def _plot_metric_bar(class_names, values, macro_avg, metric_name, color_map, edge_color, figsize, show, save_path=None):
    """Internal helper to plot a metric bar chart."""
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(class_names))
    colors = color_map(0.3 + 0.7 * values)
    bars = ax.bar(x, values, color=colors, edgecolor=edge_color, linewidth=1.2)

    ax.axhline(
        y=macro_avg,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Macro Avg: {macro_avg:.2%}',
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f'{val:.2%}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )

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
    show_saved_figure(save_path, show=show)
    plt.close(fig)

    return fig


def plot_precision(predictions, labels, class_names, figsize=(10, 6), show=True, save_path=None):
    """Plot precision for all classes as a bar chart."""
    metrics = calculate_precision_recall(predictions, labels, class_names)
    fig = _plot_metric_bar(
        class_names,
        metrics['precision'],
        metrics['macro_precision'],
        'Precision',
        plt.cm.Greens,
        'darkgreen',
        figsize,
        show,
        save_path,
    )
    return fig, metrics['precision']


def plot_recall(predictions, labels, class_names, figsize=(10, 6), show=True, save_path=None):
    """Plot recall for all classes as a bar chart."""
    metrics = calculate_precision_recall(predictions, labels, class_names)
    fig = _plot_metric_bar(
        class_names,
        metrics['recall'],
        metrics['macro_recall'],
        'Recall',
        plt.cm.Blues,
        'darkblue',
        figsize,
        show,
        save_path,
    )
    return fig, metrics['recall']


def plot_precision_recall_combined(
    predictions,
    labels,
    class_names,
    figsize=None,
    show=True,
    save_path='outputs/precision_recall_combined.png',
    sort_by='f1',
):
    """
    Plot precision and recall as a tall horizontal bar chart.

    For many classes this is much more readable than a vertical grouped bar chart.
    Classes are sorted by F1 by default so weak classes are easy to find.
    """
    metrics = calculate_precision_recall(predictions, labels, class_names)
    precision, recall, f1 = metrics['precision'], metrics['recall'], metrics['f1']

    if sort_by == 'f1':
        order = np.argsort(f1)
    elif sort_by == 'name':
        order = np.argsort(class_names)
    else:
        order = np.arange(len(class_names))

    names = [class_names[i] for i in order]
    precision = precision[order]
    recall = recall[order]
    f1 = f1[order]

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
        label='Precision',
        color='forestgreen',
        edgecolor='darkgreen',
        alpha=0.85,
    )
    ax.barh(
        y + height / 2,
        recall,
        height,
        label='Recall',
        color='steelblue',
        edgecolor='darkblue',
        alpha=0.85,
    )

    ax.axvline(
        metrics['macro_f1'],
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f"Macro F1: {metrics['macro_f1']:.2%}",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Score')
    ax.set_xlim(0, 1.05)
    ax.set_title(
        f'Precision and Recall per Class (sorted by F1, {n_classes} classes)',
        fontsize=14,
        fontweight='bold',
    )
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    plt.close(fig)

    print(f"\nPrecision and Recall Summary:")
    print(f"{'Class':<40} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 74)
    for name, prec, rec, f1_score_val in zip(names, precision, recall, f1):
        print(f"{name:<40} {prec:>10.2%} {rec:>10.2%} {f1_score_val:>10.2%}")
    print("-" * 74)
    print(
        f"{'Macro Avg':<40} "
        f"{metrics['macro_precision']:>10.2%} "
        f"{metrics['macro_recall']:>10.2%} "
        f"{metrics['macro_f1']:>10.2%}"
    )

    return fig, metrics['precision'], metrics['recall']


def save_metrics_html_report(
    predictions,
    labels,
    class_names,
    save_path='outputs/metrics_report.html',
    show=True,
):
    """
    Save a scrollable HTML report with per-class metrics.
    This is the most readable way to inspect 100+ classes.
    """
    metrics = calculate_precision_recall(predictions, labels, class_names)
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    order = np.argsort(-f1)

    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{class_names[idx]}</td>"
            f"<td>{precision[idx]:.2%}</td>"
            f"<td>{recall[idx]:.2%}</td>"
            f"<td>{f1[idx]:.2%}</td>"
            "</tr>"
        )

    overall_accuracy = (predictions == labels).sum() / max(len(labels), 1)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Classification Metrics Report</title>
  <style>
    body {{
      font-family: Segoe UI, Arial, sans-serif;
      margin: 24px;
      background: #f7f7f7;
      color: #222;
    }}
    .card {{
      background: white;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1);
      max-width: 1100px;
    }}
    h1 {{ margin-top: 0; }}
    .summary {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}
    .pill {{
      background: #eef5ff;
      border: 1px solid #c9ddff;
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 14px;
    }}
    .table-wrap {{
      max-height: 70vh;
      overflow: auto;
      border: 1px solid #ddd;
      border-radius: 6px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #eee;
      text-align: left;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f0f0f0;
      z-index: 1;
    }}
    tr:nth-child(even) {{ background: #fafafa; }}
    td:nth-child(n+3), th:nth-child(n+3) {{ text-align: right; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Classification Metrics Report</h1>
    <div class="summary">
      <div class="pill">Classes: {len(class_names)}</div>
      <div class="pill">Accuracy: {overall_accuracy:.2%}</div>
      <div class="pill">Macro Precision: {metrics['macro_precision']:.2%}</div>
      <div class="pill">Macro Recall: {metrics['macro_recall']:.2%}</div>
      <div class="pill">Macro F1: {metrics['macro_f1']:.2%}</div>
    </div>
    <p>Scroll the table below. Classes are sorted by F1 (best first).</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    return save_path


def print_classification_report(predictions, labels, class_names):
    """
    Print a detailed classification report with precision, recall, and F1-score.
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
