"""Configuration and hyperparameters for training."""

import os

import torch

# Data settings
ZIP_FILE_PATH = "src/data/archive.zip"
DATASET_DIR = "src/data/extracted"
IMAGE_SIZE = 128
BATCH_SIZE = 48
TEST_SIZE = 0.2
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5

# Model settings
CONV_CHANNELS = [3, 32, 64, 128, 256]
BLOCKS_PER_STAGE = 2
# Pool after each stage except the last to keep more spatial detail.
POOL_AFTER_STAGE = [True, True, True, False]
LEAKY_RELU = True
LEAKY_SLOPE = 0.01
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
POOL_SIZE = 2
FC_HIDDEN_SIZE = 256
DROPOUT = 0.4
BATCH_NORM = True

# DataLoader settings
NUM_WORKERS = min(8, max(4, (os.cpu_count() or 4) // 2))
PREFETCH_FACTOR = 4
PIN_MEMORY = torch.cuda.is_available()
USE_CLASS_BALANCED_SAMPLING = True

# Performance settings (do not change model architecture)
USE_AMP = torch.cuda.is_available()
USE_TTA_DURING_TRAINING = False
USE_CHANNELS_LAST = torch.cuda.is_available()
CUDNN_BENCHMARK = True
TORCH_COMPILE = False

# Training settings
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
USE_TTA = True
NUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 25
CHECKPOINT_PATH = "outputs/best_model.pt"
RANDOM_SEED = 42

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
