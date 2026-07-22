"""Data loading and transformation utilities"""

import io
import zipfile
from pathlib import Path
from PIL import Image
import requests
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

from .anime_dataset import AnimeDataset
from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, BATCH_SIZE, TEST_SIZE, RANDOM_SEED, CLASS_NAMES


def _match_class_name(file_name, class_names=CLASS_NAMES):
    """Match a zip entry to a class using folder name or filename prefix."""
    path = file_name.replace('\\', '/')
    parts = path.split('/')
    folder = parts[-2] if len(parts) >= 2 else ''
    stem = Path(parts[-1]).stem

    for class_name in class_names:
        if (
            folder == class_name
            or folder.startswith(f'{class_name}_')
            or folder.startswith(f'{class_name}(')
            or path.startswith(class_name)
            or stem.startswith(class_name)
        ):
            return class_name
    return None


def load_images_from_zip(zip_file, class_names=CLASS_NAMES):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        images = {name: [] for name in class_names}
        for file_name in zip_ref.namelist():
            lower_name = file_name.lower()
            if not lower_name.endswith(('.jpg', '.jpeg', '.png')):
                continue

            class_name = _match_class_name(file_name, class_names)
            if class_name is None:
                continue

            with zip_ref.open(file_name) as file:
                img = Image.open(file).convert('RGB')
                images[class_name].append(np.array(img))
    return images

def download_and_load_images_from_url(zip_file_url):
    response = requests.get(zip_file_url)
    zip_file_bytes = io.BytesIO(response.content)
    images = load_images_from_zip(zip_file_bytes)
    for class_name, class_images in images.items():
        print(f"Number of images of {class_name}:", len(class_images))
    return images

def download_and_load_images_from_path(zip_file_path):
    images = load_images_from_zip(zip_file_path)
    for class_name, class_images in images.items():
        print(f"Number of images of {class_name}:", len(class_images))
    return images

# Define transforms
def define_transforms(image_size=IMAGE_SIZE, normalize_mean=NORMALIZE_MEAN, normalize_std=NORMALIZE_STD):
    transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((normalize_mean, normalize_mean, normalize_mean), (normalize_std, normalize_std, normalize_std))
    ])
    return transform


def get_dataloaders(dataset, batch_size=BATCH_SIZE, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    # Generate a list of indices for the entire dataset
    indices = list(range(len(dataset)))

    # Split the indices into training and validation sets
    train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    # Create samplers for training and validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create DataLoader objects for training and validation sets
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader
