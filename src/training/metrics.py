"""Evaluation helpers: TTA, accuracy, and prediction collection."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DEVICE, USE_TTA
from src.training.performance import to_device


def predict_logits(
    model: nn.Module,
    inputs: torch.Tensor,
    use_tta: bool = USE_TTA,
) -> torch.Tensor:
    """Forward pass with optional horizontal-flip test-time augmentation."""
    logits = model(inputs)
    if not use_tta:
        return logits
    flipped = torch.flip(inputs, dims=[3])
    return (logits + model(flipped)) / 2.0


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device = DEVICE,
    use_tta: bool = False,
) -> tuple[float, float]:
    """Compute average loss and accuracy in a single pass over the loader."""
    model.eval()
    running_loss = torch.zeros((), device=device)
    correct = 0
    total = 0
    num_batches = 0

    for images, labels in data_loader:
        images = to_device(images, device)
        labels = labels.to(device, non_blocking=device.type == "cuda")
        outputs = predict_logits(model, images, use_tta=use_tta)
        running_loss += criterion(outputs, labels).detach()
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += int((predicted == labels).sum().item())
        num_batches += 1

    accuracy = correct / max(total, 1)
    return float(running_loss.item()) / max(num_batches, 1), accuracy


@torch.inference_mode()
def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device = DEVICE,
    use_tta: bool = USE_TTA,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect predicted and true labels for a loader."""
    model.eval()
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for inputs, labels in data_loader:
        inputs = to_device(inputs, device)
        outputs = predict_logits(model, inputs, use_tta=use_tta)
        all_predictions.append(outputs.argmax(dim=1).cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_predictions).numpy(),
        torch.cat(all_labels).numpy(),
    )
