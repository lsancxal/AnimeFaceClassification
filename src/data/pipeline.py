"""High-level data pipeline helpers."""

from __future__ import annotations

from torch.utils.data import DataLoader

from src.config import CACHE_IMAGES_IN_MEMORY
from src.data.anime_dataset import AnimeDataset
from src.data.dataloaders import get_dataloaders
from src.data.transforms import define_train_transforms, define_val_transforms


def build_train_val_loaders(
    samples: list[tuple[str, int]],
    class_names: list[str],
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val datasets (shared RAM cache when enabled) and DataLoaders.
    """
    train_dataset = AnimeDataset(
        samples,
        class_names,
        transform=define_train_transforms(),
        cache_in_memory=CACHE_IMAGES_IN_MEMORY,
    )
    val_dataset = AnimeDataset(
        samples,
        class_names,
        transform=define_val_transforms(),
        image_cache=train_dataset.share_image_cache(),
        cache_in_memory=False,
    )
    return get_dataloaders(train_dataset, val_dataset)
