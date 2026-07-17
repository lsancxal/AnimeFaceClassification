"""Configuration and hyperparameters for training."""

import torch

#Training settings
LARNING_RATE = 0.1
NUM_EPOCHS = 10
RANDOM_SEED = 0

#Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")