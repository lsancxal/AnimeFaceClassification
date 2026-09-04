"""Anime face classification CNN."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import (
    BLOCKS_PER_STAGE,
    CONV_CHANNELS,
    DROPOUT,
    FC_HIDDEN_SIZE,
    POOL_AFTER_STAGE,
    POOL_SIZE,
)
from src.models.blocks import ResidualBlock, make_activation


class AnimeCNN(nn.Module):
    """Residual CNN with global average pooling and a compact classifier head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        if len(POOL_AFTER_STAGE) != len(CONV_CHANNELS) - 1:
            raise ValueError(
                "POOL_AFTER_STAGE must have one entry per stage "
                f"(got {len(POOL_AFTER_STAGE)}, expected {len(CONV_CHANNELS) - 1})."
            )

        stages: list[nn.Module] = []
        for stage_idx, (in_ch, out_ch) in enumerate(
            zip(CONV_CHANNELS[:-1], CONV_CHANNELS[1:])
        ):
            blocks: list[nn.Module] = [ResidualBlock(in_ch, out_ch)]
            for _ in range(BLOCKS_PER_STAGE - 1):
                blocks.append(ResidualBlock(out_ch, out_ch))
            if POOL_AFTER_STAGE[stage_idx]:
                blocks.append(nn.MaxPool2d(POOL_SIZE, POOL_SIZE))
            stages.append(nn.Sequential(*blocks))

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Sequential(
            nn.Linear(CONV_CHANNELS[-1], FC_HIDDEN_SIZE),
            make_activation(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_HIDDEN_SIZE, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)
