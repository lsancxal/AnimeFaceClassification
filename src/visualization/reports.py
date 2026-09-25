"""End-of-run report generation."""

from __future__ import annotations

from torch.nn import Module
from torch.utils.data import DataLoader

from src.training.metrics import get_predictions
from src.training.trainer import TrainingHistory
from src.visualization.accuracy_and_cost import plot_accuracy_and_cost
from src.visualization.confusion_matrix import (
    plot_confusion_matrix,
    plot_top_confused_pairs,
)
from src.visualization.html_report import save_metrics_html_report
from src.visualization.precision_and_recall import plot_precision_recall_combined
from src.visualization.utils import plot_losses


def save_training_reports(
    history: TrainingHistory,
    model: Module,
    val_loader: DataLoader,
    class_names: list[str],
    *,
    show: bool = True,
) -> None:
    """Save loss/accuracy plots, confusion metrics, and the HTML report."""
    print("Plotting losses...")
    plot_losses(history.train_losses, history.val_losses, show=show)

    print("Plotting accuracy and cost...")
    plot_accuracy_and_cost(history.train_losses, history.accuracies, show=show)

    predictions, labels = get_predictions(model, val_loader)

    print("Plotting confusion matrix...")
    plot_confusion_matrix(predictions, labels, class_names=class_names, show=show)

    print("Plotting top confused pairs...")
    plot_top_confused_pairs(predictions, labels, class_names=class_names, show=show)

    print("Plotting precision and recall...")
    plot_precision_recall_combined(
        predictions, labels, class_names=class_names, show=show
    )

    print("Saving scrollable HTML metrics report...")
    save_metrics_html_report(predictions, labels, class_names=class_names, show=show)

    print(f"\nLast logged accuracy: {history.accuracies[-1]:.4f}")
    print(f"Peak logged accuracy: {max(history.accuracies):.4f}")
