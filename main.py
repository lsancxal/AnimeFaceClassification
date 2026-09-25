"""Anime face classification training entrypoint."""

import matplotlib

matplotlib.use("Agg")

from src.config import (
    BATCH_NORM,
    BATCH_SIZE,
    BLOCKS_PER_STAGE,
    CACHE_IMAGES_IN_MEMORY,
    CONV_CHANNELS,
    DEVICE,
    DROPOUT,
    IMAGE_SIZE,
    LEAKY_RELU,
    MIXUP_ALPHA,
    RANDOM_SEED,
    USE_AMP,
    USE_CLASS_BALANCED_SAMPLING,
    USE_TTA,
    USE_TTA_DURING_TRAINING,
    WEIGHT_DECAY,
    ZIP_FILE_PATH,
)
from src.data import build_train_val_loaders, prepare_dataset
from src.models import AnimeCNN
from src.training import Trainer, enable_performance_features, optimize_model
from src.utils import seed_everything
from src.visualization import (
    plot_random_images,
    save_training_reports,
    set_run_timestamp,
)


def _print_run_config(model: AnimeCNN) -> None:
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Cache images in memory: {CACHE_IMAGES_IN_MEMORY}")
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
    print(f"Run timestamp: {set_run_timestamp()}")

    seed_everything(RANDOM_SEED)
    enable_performance_features()
    print(f"Using device: {DEVICE}")

    print(f"Preparing dataset from: {ZIP_FILE_PATH}")
    dataset_root, samples, class_names = prepare_dataset()
    print(f"Dataset root: {dataset_root}")
    print(f"Using {len(class_names)} classes ({len(samples)} images)")

    print("Plotting random images from disk...")
    plot_random_images(samples, class_names, show=True)

    train_loader, val_loader = build_train_val_loaders(samples, class_names)
    model = optimize_model(AnimeCNN(num_classes=len(class_names)))
    _print_run_config(model)

    print("Training and evaluating model...")
    history = Trainer(model).fit(train_loader, val_loader)
    save_training_reports(history, model, val_loader, class_names, show=True)


if __name__ == "__main__":
    main()
