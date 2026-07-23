"""Data loading and transformation utilities"""

import io
import zipfile
from PIL import Image
import requests
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

from .anime_dataset import AnimeDataset
from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, BATCH_SIZE, TEST_SIZE, RANDOM_SEED


def _discover_class_names_from_zip_ref(zip_ref):
    class_names = set()
    for name in zip_ref.namelist():
        if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        parts = name.replace('\\', '/').split('/')
        if len(parts) >= 3 and parts[0] == 'dataset':
            class_names.add(parts[1])
    return sorted(class_names)


def discover_class_names_from_zip(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        return _discover_class_names_from_zip_ref(zip_ref)


# Keep old misspelled name as an alias so existing imports keep working
discover_calss_names_from_zip = discover_class_names_from_zip


def load_images_from_zip(zip_file, class_names=None):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        if class_names is None:
            class_names = _discover_class_names_from_zip_ref(zip_ref)
        images = {name: [] for name in class_names}

        for file_name in zip_ref.namelist():
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            parts = file_name.replace('\\', '/').split('/')
            if len(parts) < 3 or parts[0] != 'dataset':
                continue
            class_name = parts[1]
            if class_name not in images:
                continue
            with zip_ref.open(file_name) as f:
                img = Image.open(f).convert('RGB')
                images[class_name].append(np.array(img))
    return images, class_names


def download_and_load_images_from_url(zip_file_url):
    response = requests.get(zip_file_url)
    zip_file_bytes = io.BytesIO(response.content)
    images, class_names = load_images_from_zip(zip_file_bytes)
    for class_name, class_images in images.items():
        print(f"Number of images of {class_name}:", len(class_images))
    return images, class_names


def download_and_load_images_from_path(zip_file_path, class_names=None):
    images, class_names = load_images_from_zip(zip_file_path, class_names=class_names)
    print(f"Loaded {len(class_names)} classes, {sum(len(v) for v in images.values())} images")
    for class_name, class_images in images.items():
        print(f"Number of images of {class_name}:", len(class_images))
    return images, class_names


def define_transforms(image_size=IMAGE_SIZE, normalize_mean=NORMALIZE_MEAN, normalize_std=NORMALIZE_STD):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            (normalize_mean, normalize_mean, normalize_mean),
            (normalize_std, normalize_std, normalize_std),
        ),
    ])
    return transform


def get_dataloaders(dataset, batch_size=BATCH_SIZE, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    indices = list(range(len(dataset)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.labels,
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader
