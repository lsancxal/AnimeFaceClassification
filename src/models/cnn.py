import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    INPUT_CHANNELS,
    CONV1_OUT_CHANNELS,
    CONV2_OUT_CHANNELS,
    KERNEL_SIZE,
    STRIDE,
    PADDING,
    POOL_SIZE,
    FC_HIDDEN_SIZE,
    FEATURE_MAP_SIZE,
)


class AnimeCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimeCNN, self).__init__()
        # if num_classes is None:
        #     num_classes = len(CLASS_NAMES)

        flattened_size = CONV2_OUT_CHANNELS * FEATURE_MAP_SIZE * FEATURE_MAP_SIZE

        # padding keeps spatial size the same through each conv
        self.conv1 = nn.Conv2d(
            INPUT_CHANNELS, CONV1_OUT_CHANNELS, KERNEL_SIZE, STRIDE, padding=PADDING
        )
        self.conv2 = nn.Conv2d(
            CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, KERNEL_SIZE, STRIDE, padding=PADDING
        )
        self.pool = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        self.fc1 = nn.Linear(flattened_size, FC_HIDDEN_SIZE)
        self.fc2 = nn.Linear(FC_HIDDEN_SIZE, num_classes)
        self.flattened_size = flattened_size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
