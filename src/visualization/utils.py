import io
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import DEVICE, RANDOM_SEED
from src.visualization.display import show_saved_figure


def get_predictions(model, data_loader):
    """Get all predictions and true labels from the model."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(DEVICE)
            output = model(x)
            _, predicted = torch.max(output, 1)
            all_predictions.append(predicted.cpu())
            all_labels.append(y)

    return (
        torch.cat(all_predictions).numpy(),
        torch.cat(all_labels).numpy(),
    )


def _load_image_from_zip(zip_ref, file_name):
    with zip_ref.open(file_name) as f:
        return np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))


def plot_random_images(
    zip_file_path,
    samples,
    class_names,
    num_images=50,
    rows=5,
    cols=10,
    seed=RANDOM_SEED,
    show=True,
    save_path="outputs/random_images.png",
):
    """Plot a random sample of images from the archive without loading all images into RAM."""
    if not samples:
        raise ValueError("No images available to plot.")

    sample_size = min(num_images, len(samples), rows * cols)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(samples), size=sample_size, replace=False)
    selected_samples = [samples[i] for i in sample_indices]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(f"Random {sample_size} Images from Archive", fontsize=16)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        for ax, (file_name, label) in zip(axes, selected_samples):
            image = _load_image_from_zip(zip_ref, file_name)
            ax.imshow(image)
            ax.set_title(class_names[label], fontsize=8)
            ax.axis("off")

    for ax in axes[sample_size:]:
        ax.axis("off")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")

    show_saved_figure(save_path, show=show)
    plt.close(fig)
    plt.close("all")

    return fig


def plot_losses(
    train_losses, val_losses, show=True, save_path="outputs/training_losses.png"
):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Training Loss")
    ax.plot(val_losses, label="Validation Loss", linestyle="--")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    ax.set_title("Training and Validation Loss")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    plt.close(fig)

    return fig
