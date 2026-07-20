import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import download_and_load_images, get_dataloaders, define_transforms
from src.data.dataset import AnimeDataset
from src.visualization.utils import plot_images_from_zip, plot_losses

from src.models.cnn import AnimeCNN

from src.training import train, evaluate, calculate_accuracy

from src.config import ZIP_FILE_URL, RANDOM_SEED



# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#Download and load images
images = download_and_load_images(ZIP_FILE_URL)

plot_images_from_zip(images)


transform = define_transforms()
# Load dataset
dataset = AnimeDataset(images, transform=transform, classes=['anastasia', 'takao'])

# Get dataloaders
train_loader, val_loader = get_dataloaders(dataset)

# Instantiate the model
model = AnimeCNN()

print(model)

input_tensor = torch.randn(1, 3, 64, 64)

def print_size(module, input, output):
    print(f"{module.__class__.__name__} output size: {output.size()}")

# Register hooks
hooks = []
for layer in model.children():
    hook = layer.register_forward_hook(print_size)
    hooks.append(hook)

# Inspect output sizes
with torch.no_grad():
    output = model(input_tensor)
print("Final output size:", output.size())

# Remove hooks
for hook in hooks:
    hook.remove()

train_losses = train(model, train_loader)
val_losses = evaluate(model, val_loader)

print('Finished Training')

plot_losses(train_losses, val_losses)

calculate_accuracy(model, val_loader)