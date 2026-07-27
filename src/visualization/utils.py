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
    """
    Get all predictions and true labels from the model.

    Args:
        model: The trained neural network model.
        data_loader: DataLoader for the dataset.

    Returns:
        tuple: (predictions, labels) as numpy arrays.
    """
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
        torch.cat(all_labels).numpy()
    )


# Plot images from the zip file
def plot_images(images, title):
    fig, axes = plt.subplots(5, 10, figsize=(10, 5))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_images_from_zip(images):
    for class_name, class_images in images.items():
        if not class_images:
            print(f"Skipping plot for '{class_name}': no images found")
            continue
        plot_images(class_images, f'{class_name} Images')


def _load_image_from_zip(zip_ref, file_name):
    with zip_ref.open(file_name) as f:
        return np.array(Image.open(io.BytesIO(f.read())).convert('RGB'))


def plot_random_images(
    zip_file_path,
    samples,
    class_names,
    num_images=50,
    rows=5,
    cols=10,
    seed=RANDOM_SEED,
    show=True,
    save_path='outputs/random_images.png',
):
    """
    Plot a random sample of images from the archive without loading all images into RAM.

    Args:
        zip_file_path: Path to the ZIP archive.
        samples: List of (file_name, label_index) tuples.
        class_names: List of class names indexed by label.
        num_images: Number of random images to plot.
        rows: Number of subplot rows.
        cols: Number of subplot columns.
        seed: Random seed for reproducible sampling.
        show: Whether to display the figure.
        save_path: Path to save the figure. Pass None to skip saving.

    Returns:
        fig: The matplotlib figure.
    """
    if not samples:
        raise ValueError("No images available to plot.")

    sample_size = min(num_images, len(samples), rows * cols)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(samples), size=sample_size, replace=False)
    selected_samples = [samples[i] for i in sample_indices]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(f'Random {sample_size} Images from Archive', fontsize=16)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for ax, (file_name, label) in zip(axes, selected_samples):
            image = _load_image_from_zip(zip_ref, file_name)
            ax.imshow(image)
            ax.set_title(class_names[label], fontsize=8)
            ax.axis('off')

    for ax in axes[sample_size:]:
        ax.axis('off')

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")

    show_saved_figure(save_path, show=show)
    plt.close(fig)
    plt.close('all')

    return fig


# Plotting the training and validation loss
def plot_losses(train_losses, val_losses, show=True, save_path='outputs/training_losses.png'):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss', linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    ax.set_title('Training and Validation Loss')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")
    show_saved_figure(save_path, show=show)
    plt.close(fig)

    return fig
