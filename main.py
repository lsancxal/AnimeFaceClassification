import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import download_and_load_images, get_dataloaders, define_transforms
from src.data.anime_dataset import AnimeDataset
from src.visualization.utils import plot_images_from_zip, plot_losses, get_predictions
from src.visualization.accuracy_and_cost import plot_accuracy_and_cost
from src.visualization.confusion_matrix import plot_confusion_matrix, plot_confusion_matrix_with_stats
from src.visualization.precision_and_recall import plot_precision, plot_recall, plot_precision_recall_combined, print_classification_report

from src.models.cnn import AnimeCNN

from src.training import train_and_evaluate

from src.config import ZIP_FILE_URL, RANDOM_SEED, DEVICE, CLASS_NAMES


def main():

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(f"Using device:{DEVICE}")

    print("Downloading and loading images...")
    #Download and load images
    images = download_and_load_images(ZIP_FILE_URL)

    print("Plotting images...")
    plot_images_from_zip(images)

    print("Defining transforms...")
    transform = define_transforms()
    
    print("Loading dataset...")
    dataset = AnimeDataset(images, transform=transform, classes=CLASS_NAMES)

    print("Getting dataloaders...")
    train_loader, val_loader = get_dataloaders(dataset)

    print("Instantiating model...")
    model = AnimeCNN()
    print(f"Model architecture: \n{model}\n")

    # input_tensor = torch.randn(1, 3, 64, 64)
    # def print_size(module, input, output):
    #     print(f"{module.__class__.__name__} output size: {output.size()}")

    # # Register hooks
    # hooks = []
    # for layer in model.children():
    #     hook = layer.register_forward_hook(print_size)
    #     hooks.append(hook)

    # # Inspect output sizes
    # with torch.no_grad():
    #     output = model(input_tensor)
    # print("Final output size:", output.size())

    # # Remove hooks
    # for hook in hooks:
    #     hook.remove()

    print("Training and evaluating model...")
    train_losses, val_losses, accuracy_list = train_and_evaluate(model, train_loader, val_loader)

    print("Plotting losses...")
    plot_losses(train_losses, val_losses)

    print("Plotting accuracy and cost...")
    plot_accuracy_and_cost(train_losses, accuracy_list)

    predictions, labels = get_predictions(model, val_loader)

    print("Plotting confusion matrix...")
    plot_confusion_matrix(predictions, labels)

    print("Plotting precision and recall...")
    plot_precision(predictions, labels)
    plot_recall(predictions, labels)
    plot_precision_recall_combined(predictions, labels)
    print_classification_report(predictions, labels)

    print(f"\nFinal accuracy: {accuracy_list[-1]:.4f}")

if __name__ == "__main__":
    main()