import io
import zipfile

from PIL import Image
from torch.utils.data import Dataset


class AnimeDataset(Dataset):
    def __init__(self, zip_file_path, samples, class_names, transform=None):
        self.zip_file_path = zip_file_path
        self.samples = samples
        self.class_names = class_names
        self.transform = transform
        self.labels = [label for _, label in samples]
        self._zip_ref = None

    def _get_zip_ref(self):
        # Reuse one ZipFile handle per worker instead of reopening every sample.
        if self._zip_ref is None:
            self._zip_ref = zipfile.ZipFile(self.zip_file_path, "r")
        return self._zip_ref

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, label = self.samples[idx]
        zip_ref = self._get_zip_ref()

        with zip_ref.open(file_name) as f:
            image = Image.open(io.BytesIO(f.read())).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
