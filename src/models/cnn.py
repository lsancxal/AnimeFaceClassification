import torch.nn as nn

from src.config import (
    CONV_CHANNELS,
    KERNEL_SIZE,
    STRIDE,
    PADDING,
    POOL_SIZE,
    IMAGE_SIZE,
    FC_HIDDEN_SIZE,
    LEAKY_SLOPE,
    BATCH_NORM,
    LEAKY_RELU,
)


def _activation():
    if LEAKY_RELU:
        return nn.LeakyReLU(LEAKY_SLOPE)
    return nn.ReLU()


class AnimeCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimeCNN, self).__init__()

        conv_layers = []
        for in_ch, out_ch in zip(CONV_CHANNELS[:-1], CONV_CHANNELS[1:]):
            conv_layers.append(
                nn.Conv2d(in_ch, out_ch, KERNEL_SIZE, STRIDE, padding=PADDING)
            )
            if BATCH_NORM:
                conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(_activation())
            conv_layers.append(nn.MaxPool2d(POOL_SIZE, POOL_SIZE))

        self.features = nn.Sequential(*conv_layers)

        num_pools = len(CONV_CHANNELS) - 1
        feature_map_size = IMAGE_SIZE // (POOL_SIZE**num_pools)
        flattened_size = CONV_CHANNELS[-1] * feature_map_size * feature_map_size

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, FC_HIDDEN_SIZE),
            _activation(),
            nn.Linear(FC_HIDDEN_SIZE, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
