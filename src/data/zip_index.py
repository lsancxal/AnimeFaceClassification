"""Extract ZIP archives to disk and index image samples."""

from __future__ import annotations

import os
import zipfile

from src.config import DATASET_DIR, ZIP_FILE_PATH

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _is_dataset_image(member_name: str) -> bool:
    normalized = member_name.replace("\\", "/")
    if not normalized.lower().endswith(IMAGE_EXTENSIONS):
        return False
    parts = normalized.split("/")
    return len(parts) >= 3 and parts[0] == "dataset"


def extract_zip_to_disk(
    zip_file_path: str = ZIP_FILE_PATH,
    extract_dir: str = DATASET_DIR,
    force: bool = False,
) -> str:
    """
    Extract the archive to disk once and reuse it on later runs.

    Returns the absolute path to the extracted dataset root
    (``extract_dir/dataset``).
    """
    dataset_root = os.path.join(extract_dir, "dataset")
    marker_path = os.path.join(extract_dir, ".extract_complete")

    if (
        not force
        and os.path.isdir(dataset_root)
        and os.path.isfile(marker_path)
    ):
        print(f"Using extracted dataset at: {dataset_root}")
        return dataset_root

    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting {zip_file_path} -> {extract_dir} ...")

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        members = [name for name in zip_ref.namelist() if _is_dataset_image(name)]
        for index, member in enumerate(members, start=1):
            zip_ref.extract(member, extract_dir)
            if index % 1000 == 0 or index == len(members):
                print(f"  Extracted {index}/{len(members)} images")

    with open(marker_path, "w", encoding="utf-8") as handle:
        handle.write(os.path.abspath(zip_file_path))

    print(f"Extraction complete: {dataset_root}")
    return dataset_root


def build_samples_from_directory(
    dataset_root: str,
) -> tuple[list[tuple[str, int]], list[str]]:
    """
    Index image paths and class labels from an extracted folder.

    Expected layout: ``<dataset_root>/<class_name>/<image>``.
    Returns ``(samples, class_names)`` where each sample is
    ``(absolute_path, label_idx)``.
    """
    class_names: set[str] = set()
    image_files: list[tuple[str, str]] = []

    for class_name in os.listdir(dataset_root):
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_names.add(class_name)
        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            image_files.append((os.path.join(class_dir, file_name), class_name))

    sorted_class_names = sorted(class_names)
    class_to_index = {name: idx for idx, name in enumerate(sorted_class_names)}
    samples = [
        (file_path, class_to_index[class_name])
        for file_path, class_name in image_files
    ]
    return samples, sorted_class_names


def prepare_dataset(
    zip_file_path: str = ZIP_FILE_PATH,
    extract_dir: str = DATASET_DIR,
    force: bool = False,
) -> tuple[str, list[tuple[str, int]], list[str]]:
    """
    Ensure the ZIP is extracted, then return ``(dataset_root, samples, class_names)``.
    """
    dataset_root = extract_zip_to_disk(
        zip_file_path=zip_file_path,
        extract_dir=extract_dir,
        force=force,
    )
    samples, class_names = build_samples_from_directory(dataset_root)
    return dataset_root, samples, class_names


def build_samples_from_zip(
    zip_file_path: str,
) -> tuple[list[tuple[str, int]], list[str]]:
    """
    Backward-compatible helper: extract ZIP if needed, then index from disk.
    """
    _, samples, class_names = prepare_dataset(zip_file_path=zip_file_path)
    return samples, class_names
