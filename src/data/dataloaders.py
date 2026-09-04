"""Train / validation DataLoader construction."""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from src.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    RANDOM_SEED,
    TEST_SIZE,
    USE_CLASS_BALANCED_SAMPLING,
)


def _loader_kwargs(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    *,
    drop_last: bool = False,
) -> dict:
    kwargs: dict = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs.update(
            {
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": PREFETCH_FACTOR,
            }
        )
    return kwargs


def get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    batch_size: int = BATCH_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    use_class_balanced_sampling: bool = USE_CLASS_BALANCED_SAMPLING,
) -> tuple[DataLoader, DataLoader]:
    """
    Build stratified train/val loaders.

    Pass two datasets (same samples, different transforms) for augmentation.
    If only one dataset is given, it is used for both splits.
    """
    from sklearn.model_selection import train_test_split

    if val_dataset is None:
        val_dataset = train_dataset

    labels = getattr(train_dataset, "labels", None)
    if labels is None:
        raise AttributeError("train_dataset must expose a `.labels` attribute.")

    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    train_kwargs = _loader_kwargs(batch_size, num_workers, pin_memory, drop_last=True)
    val_kwargs = _loader_kwargs(batch_size, num_workers, pin_memory)

    if use_class_balanced_sampling:
        train_labels = np.asarray([labels[i] for i in train_indices], dtype=np.int64)
        class_counts = np.bincount(train_labels)
        class_counts[class_counts == 0] = 1
        sample_weights = (1.0 / class_counts)[train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_subset, sampler=train_sampler, **train_kwargs)
    else:
        train_loader = DataLoader(
            train_subset,
            sampler=SubsetRandomSampler(range(len(train_subset))),
            **train_kwargs,
        )

    val_loader = DataLoader(val_subset, shuffle=False, **val_kwargs)
    return train_loader, val_loader
