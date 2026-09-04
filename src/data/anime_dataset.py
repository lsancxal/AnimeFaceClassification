"""Filesystem-backed image dataset."""

from __future__ import annotations

from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset


class AnimeDataset(Dataset):
    """Load RGB images on demand from disk paths."""

    def __init__(
        self,
        samples: Sequence[tuple[str, int]],
        class_names: Sequence[str],
        transform: Callable | None = None,
    ) -> None:
        self.samples = list(samples)
        self.class_names = list(class_names)
        self.transform = transform
        self.labels = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        file_path, label = self.samples[idx]
        with Image.open(file_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label
