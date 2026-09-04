"""Shared helpers for saving and opening figures."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.visualization.display import show_saved_figure
from src.visualization.paths import add_run_timestamp


def save_and_show_figure(
    fig: Figure,
    save_path: str,
    *,
    show: bool = True,
    dpi: int = 150,
    add_timestamp: bool = True,
) -> None:
    if add_timestamp:
        add_run_timestamp(fig)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"  Saved: {save_path}")

    show_saved_figure(save_path, show=show)
    plt.close(fig)
