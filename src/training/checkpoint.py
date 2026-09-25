"""Checkpoint save / load helpers."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from src.config import DEVICE
from src.training.performance import unwrap_model


def save_checkpoint(
    path: str,
    model: nn.Module,
    *,
    epoch: int,
    val_accuracy: float,
    val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap_model(model).state_dict(),
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    *,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
