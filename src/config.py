"""Configuration and hyperparameters for training."""

import torch

#Data settings
ZIP_FILE_URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/xZQHOyN8ONT92kH-ASb4Pw/data.zip'
ZIP_FILE_PATH = 'src/data/archive.zip'
IMAGE_SIZE = 64
BATCH_SIZE = 8
TEST_SIZE = 0.2
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5
CLASS_NAMES = ['kirito', 'zero_two', 'sinon',  'raphtalia']

#Model settings
INPUT_CHANNELS = 3
CONV1_OUT_CHANNELS = 32
CONV2_OUT_CHANNELS = 64
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
POOL_SIZE = 2
FC_HIDDEN_SIZE = 128
# After two MaxPool(2) layers, spatial size is IMAGE_SIZE / 4
FEATURE_MAP_SIZE = IMAGE_SIZE // (POOL_SIZE ** 2)

#Training settings
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
RANDOM_SEED = 42

#Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
