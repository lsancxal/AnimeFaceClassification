import os
import matplotlib.pyplot as plt
from sympy import plot
import torch
from src.config import DEVICE

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
    # Plot images from 'anastasia'
    plot_images(images['anastasia'], 'Anastasia Images')
    # Plot images from 'takao'
    plot_images(images['takao'], 'Takao Images')

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