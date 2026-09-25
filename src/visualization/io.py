"""Shared helpers for saving and opening figures."""

from __future__ import annotations

import os
import tempfile
import time

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
        absolute_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(absolute_path) or ".", exist_ok=True)
        saved_path = _save_figure_resilient(fig, absolute_path, dpi=dpi)
        print(f"  Saved: {saved_path}")
        save_path = saved_path

    show_saved_figure(save_path, show=show)
    plt.close(fig)


def _save_figure_resilient(fig: Figure, absolute_path: str, *, dpi: int) -> str:
    """
    Save a figure robustly on Windows.

    - Writes through a temp file
    - Retries without bbox_inches='tight' if needed
    - If the destination is locked by a viewer, falls back to a unique name
    """
    directory = os.path.dirname(absolute_path) or "."
    stem, ext = os.path.splitext(absolute_path)

    fd, temp_path = tempfile.mkstemp(suffix=ext or ".png", dir=directory)
    os.close(fd)

    try:
        try:
            fig.savefig(temp_path, bbox_inches="tight", dpi=dpi)
        except (OSError, ValueError):
            fig.savefig(temp_path, dpi=dpi)

        for candidate in (
            absolute_path,
            f"{stem}_{int(time.time())}{ext or '.png'}",
        ):
            try:
                if os.path.exists(candidate):
                    try:
                        os.remove(candidate)
                    except OSError:
                        if candidate == absolute_path:
                            continue
                        raise
                os.replace(temp_path, candidate)
                return candidate
            except OSError:
                continue

        # Last resort: keep the temp file name.
        return temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
