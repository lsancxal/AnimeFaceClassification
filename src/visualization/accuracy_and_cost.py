"""Training progress plots."""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.visualization.io import save_and_show_figure
from src.visualization.paths import output_path


def plot_accuracy_and_cost(
    cost_list: list[float],
    accuracy_list: list[float],
    show: bool = True,
    save_path: str | None = None,
):
    """Plot training cost and accuracy over epochs."""
    if save_path is None:
        save_path = output_path("accuracy_and_cost.png")

    if len(cost_list) != len(accuracy_list):
        raise ValueError(
            f"cost_list and accuracy_list must have the same length "
            f"({len(cost_list)} vs {len(accuracy_list)})."
        )

    epochs = range(1, len(cost_list) + 1)
    accuracy_percent = [acc * 100 for acc in accuracy_list]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    (cost_line,) = ax1.plot(
        epochs, cost_list, color="tab:red", marker="o", label="Cost"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cost", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    (accuracy_line,) = ax2.plot(
        epochs,
        accuracy_percent,
        color="tab:blue",
        marker="o",
        label="Accuracy",
    )
    ax2.set_ylabel("Accuracy (%)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0, 105)

    fig.suptitle("Training Progress", fontsize=14, y=0.98)
    fig.legend(
        [cost_line, accuracy_line],
        ["Cost", "Accuracy"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_and_show_figure(fig, save_path, show=show)
    return fig
