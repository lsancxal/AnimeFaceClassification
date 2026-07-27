import matplotlib
matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import download_and_load_images_from_url, download_and_load_images_from_path, build_samples_from_zip, get_dataloaders, define_transforms
from src.data.anime_dataset import AnimeDataset
from src.visualization.utils import plot_images_from_zip, plot_random_images, plot_losses, get_predictions
from src.visualization.accuracy_and_cost import plot_accuracy_and_cost
from src.visualization.confusion_matrix import plot_confusion_matrix, plot_confusion_matrix_with_stats
from src.visualization.precision_and_recall import plot_precision, plot_recall, plot_precision_recall_combined, print_classification_report

from src.models.cnn import AnimeCNN

from src.training import train_and_evaluate

from src.config import ZIP_FILE_URL, RANDOM_SEED, DEVICE, ZIP_FILE_PATH


def main():

    print("Starting...")

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(f"Using device:{DEVICE}")

    # print("Downloading and loading images...")
    # images = download_and_load_images_from_url(ZIP_FILE_URL)

    print("Loading images from path...")
    samples, class_names = build_samples_from_zip(ZIP_FILE_PATH)
    print(f"Using {len(class_names)} classes")

    print("Plotting random images from archive...")
    plot_random_images(ZIP_FILE_PATH, samples, class_names, show=True)

    print("Defining transforms...")
    transform = define_transforms()
    
    print("Loading dataset...")
    dataset = AnimeDataset(ZIP_FILE_PATH, samples, class_names, transform=transform)

    print("Getting dataloaders...")
    train_loader, val_loader = get_dataloaders(dataset)

    print("Instantiating model...")
    model = AnimeCNN(num_classes=len(class_names))
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
    plot_confusion_matrix(predictions, labels, class_names=class_names)

    print("Plotting precision and recall...")
    plot_precision_recall_combined(predictions, labels, class_names=class_names)
    print_classification_report(predictions, labels, class_names=class_names)

    print(f"\nFinal accuracy: {accuracy_list[-1]:.4f}")

if __name__ == "__main__":
    main()