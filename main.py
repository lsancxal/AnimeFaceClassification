"""Anime face classification training entrypoint."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from src.config import (
    BATCH_NORM,
    BLOCKS_PER_STAGE,
    CONV_CHANNELS,
    DEVICE,
    DROPOUT,
    IMAGE_SIZE,
    LEAKY_RELU,
    MIXUP_ALPHA,
    RANDOM_SEED,
    USE_CLASS_BALANCED_SAMPLING,
    USE_AMP,
    USE_TTA,
    USE_TTA_DURING_TRAINING,
    WEIGHT_DECAY,
    ZIP_FILE_PATH,
)
from src.data import (
    AnimeDataset,
    define_train_transforms,
    define_val_transforms,
    get_dataloaders,
    prepare_dataset,
)
from src.models import AnimeCNN
from src.training import Trainer, get_predictions
from src.training.performance import enable_performance_features, optimize_model
from src.visualization import (
    plot_accuracy_and_cost,
    plot_confusion_matrix,
    plot_losses,
    plot_precision_recall_combined,
    plot_random_images,
    plot_top_confused_pairs,
    save_metrics_html_report,
    set_run_timestamp,
)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_run_config(model: AnimeCNN) -> None:
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Conv channels: {CONV_CHANNELS} ({BLOCKS_PER_STAGE} residual blocks/stage)")
    print(f"Using batch normalization: {BATCH_NORM}")
    print(f"Using leaky ReLU: {LEAKY_RELU}")
    print(f"Using dropout: {DROPOUT}")
    print(f"Using weight decay: {WEIGHT_DECAY}")
    print(f"Using mixup alpha: {MIXUP_ALPHA}")
    print(f"Using AMP: {USE_AMP}")
    print(f"Using TTA during training: {USE_TTA_DURING_TRAINING}")
    print(f"Using TTA for final reports: {USE_TTA}")
    print(f"Using class-balanced sampling: {USE_CLASS_BALANCED_SAMPLING}")
    print(f"Model architecture:\n{model}\n")


def main() -> None:
    print("Starting...")
    run_stamp = set_run_timestamp()
    print(f"Run timestamp: {run_stamp}")

    _seed_everything(RANDOM_SEED)
    enable_performance_features()
    print(f"Using device: {DEVICE}")

    print(f"Preparing dataset from: {ZIP_FILE_PATH}")
    dataset_root, samples, class_names = prepare_dataset()
    print(f"Dataset root: {dataset_root}")
    print(f"Using {len(class_names)} classes ({len(samples)} images)")

    print("Plotting random images from disk...")
    plot_random_images(samples, class_names, show=True)

    train_dataset = AnimeDataset(
        samples,
        class_names,
        transform=define_train_transforms(),
    )
    val_dataset = AnimeDataset(
        samples,
        class_names,
        transform=define_val_transforms(),
    )
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

    model = optimize_model(AnimeCNN(num_classes=len(class_names)))
    _print_run_config(model)

    print("Training and evaluating model...")
    history = Trainer(model).fit(train_loader, val_loader)

    print("Plotting losses...")
    plot_losses(history.train_losses, history.val_losses)

    print("Plotting accuracy and cost...")
    plot_accuracy_and_cost(history.train_losses, history.accuracies)

    predictions, labels = get_predictions(model, val_loader)

    print("Plotting confusion matrix...")
    plot_confusion_matrix(predictions, labels, class_names=class_names)

    print("Plotting top confused pairs...")
    plot_top_confused_pairs(predictions, labels, class_names=class_names)

    print("Plotting precision and recall...")
    plot_precision_recall_combined(predictions, labels, class_names=class_names)

    print("Saving scrollable HTML metrics report...")
    save_metrics_html_report(predictions, labels, class_names=class_names)

    print(f"\nLast logged accuracy: {history.accuracies[-1]:.4f}")
    print(f"Peak logged accuracy: {max(history.accuracies):.4f}")


if __name__ == "__main__":
    main()
