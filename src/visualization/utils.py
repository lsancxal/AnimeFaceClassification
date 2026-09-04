"""Dataset preview and training-curve plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import RANDOM_SEED
from src.visualization.io import save_and_show_figure
from src.visualization.paths import output_path


def plot_random_images(
    samples: list[tuple[str, int]],
    class_names: list[str],
    num_images: int = 50,
    rows: int = 5,
    cols: int = 10,
    seed: int = RANDOM_SEED,
    show: bool = True,
    save_path: str | None = None,
):
    """Plot a random sample of images from disk without loading all into RAM."""
    if save_path is None:
        save_path = output_path("random_images.png")

    if not samples:
        raise ValueError("No images available to plot.")

    sample_size = min(num_images, len(samples), rows * cols)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(samples), size=sample_size, replace=False)
    selected_samples = [samples[i] for i in sample_indices]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(f"Random {sample_size} Images from Dataset", fontsize=16)

    for ax, (file_path, label) in zip(axes, selected_samples):
        with Image.open(file_path) as image:
            ax.imshow(np.array(image.convert("RGB")))
        ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")

    for ax in axes[sample_size:]:
        ax.axis("off")

    fig.tight_layout()
    save_and_show_figure(fig, save_path, show=show)
    plt.close("all")
    return fig


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    show: bool = True,
    save_path: str | None = None,
):
    if save_path is None:
        save_path = output_path("training_losses.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Training Loss")
    ax.plot(val_losses, label="Validation Loss", linestyle="--")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    ax.set_title("Training and Validation Loss")
    fig.tight_layout()
    save_and_show_figure(fig, save_path, show=show)
    return fig
