"""Filesystem-backed image dataset with optional RAM cache."""

from __future__ import annotations

from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from src.config import CACHE_IMAGES_IN_MEMORY, IMAGE_SIZE


class AnimeDataset(Dataset):
    """Load RGB images from disk, optionally caching them in RAM once."""

    def __init__(
        self,
        samples: Sequence[tuple[str, int]],
        class_names: Sequence[str],
        transform: Callable | None = None,
        *,
        cache_in_memory: bool = CACHE_IMAGES_IN_MEMORY,
        image_cache: list[Image.Image] | None = None,
        cache_image_size: int | None = IMAGE_SIZE,
    ) -> None:
        self.samples = list(samples)
        self.class_names = list(class_names)
        self.transform = transform
        self.labels = [label for _, label in self.samples]
        self.cache_image_size = cache_image_size

        if image_cache is not None:
            if len(image_cache) != len(self.samples):
                raise ValueError("image_cache length must match samples length.")
            self._images = image_cache
        elif cache_in_memory:
            self._images = self._load_all_images()
        else:
            self._images = None

    def _load_all_images(self) -> list[Image.Image]:
        images: list[Image.Image] = []
        total = len(self.samples)
        size = self.cache_image_size
        print(
            f"Caching {total} images in memory"
            + (f" at {size}x{size}" if size else "")
            + "..."
        )
        for index, (file_path, _) in enumerate(self.samples, start=1):
            with Image.open(file_path) as image:
                rgb = image.convert("RGB")
                if size is not None and rgb.size != (size, size):
                    rgb = rgb.resize((size, size), Image.BILINEAR)
                images.append(rgb.copy())
            if index % 2000 == 0 or index == total:
                print(f"  Cached {index}/{total}")
        return images

    def share_image_cache(self) -> list[Image.Image] | None:
        """Return the RAM cache so train/val datasets can share one copy."""
        return self._images

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        label = self.samples[idx][1]
        if self._images is not None:
            image = self._images[idx]
        else:
            file_path, _ = self.samples[idx]
            with Image.open(file_path) as handle:
                image = handle.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label
