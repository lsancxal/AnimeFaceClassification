"""Reusable CNN building blocks."""

from __future__ import annotations

import torch.nn as nn

from src.config import (
    BATCH_NORM,
    KERNEL_SIZE,
    LEAKY_RELU,
    LEAKY_SLOPE,
    PADDING,
    STRIDE,
)


def make_activation() -> nn.Module:
    if LEAKY_RELU:
        return nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
    return nn.ReLU(inplace=True)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        bias = not BATCH_NORM

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if BATCH_NORM else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if BATCH_NORM else nn.Identity()
        self.act = make_activation()

        if in_channels != out_channels:
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            ]
            if BATCH_NORM:
                layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*layers)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)
