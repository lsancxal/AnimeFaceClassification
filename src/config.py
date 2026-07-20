"""Configuration and hyperparameters for training."""

import torch

#Data settings
ZIP_FILE_URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/xZQHOyN8ONT92kH-ASb4Pw/data.zip'
IMAGE_SIZE = 64
BATCH_SIZE = 8
TEST_SIZE = 0.2
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5

#Training settings
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
RANDOM_SEED = 42

#Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")