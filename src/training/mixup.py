"""Mixup regularization helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import MIXUP_ALPHA


def mixup_data(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = MIXUP_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup to a batch. Returns mixed inputs, paired labels, and lambda."""
    if alpha <= 0:
        return inputs, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed_inputs, labels, labels[index], lam


def mixup_criterion(
    criterion: nn.Module,
    outputs: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(
        outputs, labels_b
    )
