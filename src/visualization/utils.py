import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.config import DEVICE, RANDOM_SEED


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


def plot_random_images(
    images,
    num_images=50,
    rows=5,
    cols=10,
    seed=RANDOM_SEED,
    show=True,
    save_path=None,
):
    """
    Plot a random sample of images from the loaded archive.

    Args:
        images: Dict of class_name -> list of image arrays.
        num_images: Number of random images to plot.
        rows: Number of subplot rows.
        cols: Number of subplot columns.
        seed: Random seed for reproducible sampling.
        show: Whether to call plt.show().
        save_path: Path to save the figure. Pass None to skip saving.

    Returns:
        fig: The matplotlib figure.
    """
    labeled_images = [
        (class_name, img)
        for class_name, class_images in images.items()
        for img in class_images
    ]
    if not labeled_images:
        raise ValueError("No images available to plot.")

    sample_size = min(num_images, len(labeled_images), rows * cols)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(labeled_images), size=sample_size, replace=False)
    sample = [labeled_images[i] for i in sample_indices]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(f'Random {sample_size} Images from Archive', fontsize=16)

    for ax, (class_name, img) in zip(axes, sample):
        ax.imshow(img)
        ax.set_title(class_name, fontsize=8)
        ax.axis('off')

    for ax in axes[sample_size:]:
        ax.axis('off')

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")
    if show:
        plt.show()

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
    if show:
        plt.show()

    return fig
