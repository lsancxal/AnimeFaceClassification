"""HTML metrics report generation."""

from __future__ import annotations

import os

import numpy as np

from src.config import BATCH_NORM, LEAKY_RELU, LEAKY_SLOPE
from src.visualization.display import show_saved_figure
from src.visualization.paths import get_run_timestamp, output_path
from src.visualization.precision_and_recall import calculate_precision_recall


def save_metrics_html_report(
    predictions,
    labels,
    class_names,
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """Save a scrollable HTML report with per-class metrics."""
    if save_path is None:
        save_path = output_path("metrics_report.html")

    metrics = calculate_precision_recall(predictions, labels, class_names)
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
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
    run_stamp = get_run_timestamp()
    batch_norm_label = "Yes" if BATCH_NORM else "No"
    leaky_relu_label = f"Yes (slope={LEAKY_SLOPE})" if LEAKY_RELU else "No"

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
    h1 {{ margin-top: 0; margin-bottom: 4px; }}
    .run-stamp {{
      margin: 0 0 16px 0;
      font-size: 16px;
      font-weight: 600;
      color: #222;
    }}
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
    <p class="run-stamp">Run: {run_stamp}</p>
    <div class="summary">
      <div class="pill">Batch Norm: {batch_norm_label}</div>
      <div class="pill">Leaky ReLU: {leaky_relu_label}</div>
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

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    return save_path
