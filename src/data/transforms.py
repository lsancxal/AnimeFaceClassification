"""Image transforms for training and validation."""

from __future__ import annotations

from torchvision import transforms

from src.config import CACHE_IMAGES_IN_MEMORY, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def _normalize(normalize_mean: float, normalize_std: float) -> transforms.Normalize:
    return transforms.Normalize(
        (normalize_mean, normalize_mean, normalize_mean),
        (normalize_std, normalize_std, normalize_std),
    )


def define_train_transforms(
    image_size: int = IMAGE_SIZE,
    normalize_mean: float = NORMALIZE_MEAN,
    normalize_std: float = NORMALIZE_STD,
) -> transforms.Compose:
    """Augmented transforms for training only."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.1, hue=0.02
            ),
            transforms.ToTensor(),
            _normalize(normalize_mean, normalize_std),
        ]
    )


def define_val_transforms(
    image_size: int = IMAGE_SIZE,
    normalize_mean: float = NORMALIZE_MEAN,
    normalize_std: float = NORMALIZE_STD,
    *,
    skip_resize: bool | None = None,
) -> transforms.Compose:
    """Deterministic transforms for validation / evaluation."""
    if skip_resize is None:
        skip_resize = CACHE_IMAGES_IN_MEMORY

    ops: list = []
    if not skip_resize:
        ops.append(transforms.Resize((image_size, image_size)))
    ops.extend(
        [
            transforms.ToTensor(),
            _normalize(normalize_mean, normalize_std),
        ]
    )
    return transforms.Compose(ops)
