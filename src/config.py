"""Configuration and hyperparameters for training."""

import torch

# Data settings
ZIP_FILE_PATH = "src/data/archive.zip"
IMAGE_SIZE = 64
BATCH_SIZE = 64
TEST_SIZE = 0.2
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5

# Model settings
CONV_CHANNELS = [3, 32, 64]
LEAKY_RELU = True
LEAKY_SLOPE = 0.01
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
POOL_SIZE = 2
FC_HIDDEN_SIZE = 128
BATCH_NORM = True

# DataLoader settings
NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()

# Training settings
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
RANDOM_SEED = 42

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
