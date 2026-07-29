"""Data loading and transformation utilities"""

import zipfile

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

from src.config import (
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    BATCH_SIZE,
    TEST_SIZE,
    RANDOM_SEED,
    NUM_WORKERS,
    PIN_MEMORY,
)


def build_samples_from_zip(zip_file_path):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        class_names = set()
        image_files = []

        for file_name in zip_ref.namelist():
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            parts = file_name.replace("\\", "/").split("/")
            if len(parts) < 3 or parts[0] != "dataset":
                continue

            class_name = parts[1]
            class_names.add(class_name)
            image_files.append((file_name, class_name))

        class_names = sorted(class_names)
        class_to_index = {name: idx for idx, name in enumerate(class_names)}
        samples = [
            (file_name, class_to_index[class_name])
            for file_name, class_name in image_files
        ]

        return samples, class_names


def define_transforms(
    image_size=IMAGE_SIZE,
    normalize_mean=NORMALIZE_MEAN,
    normalize_std=NORMALIZE_STD,
):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (normalize_mean, normalize_mean, normalize_mean),
                (normalize_std, normalize_std, normalize_std),
            ),
        ]
    )


def get_dataloaders(
    dataset,
    batch_size=BATCH_SIZE,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
):
    indices = list(range(len(dataset)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.labels,
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs.update(
            {
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )

    train_loader = DataLoader(dataset, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(dataset, sampler=val_sampler, **loader_kwargs)

    return train_loader, val_loader
