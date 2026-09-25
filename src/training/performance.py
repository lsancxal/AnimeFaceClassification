"""Runtime performance helpers for faster training."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import (
    ALLOW_TF32,
    CUDNN_BENCHMARK,
    DEVICE,
    TORCH_COMPILE,
    USE_AMP,
    USE_CHANNELS_LAST,
)


def enable_performance_features() -> None:
    """Apply global PyTorch performance settings once at startup."""
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def optimize_model(model: nn.Module) -> nn.Module:
    """Apply optional runtime optimizations that do not change model weights."""
    if USE_CHANNELS_LAST and DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model


def to_device(
    tensor: torch.Tensor,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Move a batch tensor to the training device with optional channels_last."""
    kwargs: dict = {"non_blocking": device.type == "cuda"}
    if USE_CHANNELS_LAST and device.type == "cuda" and tensor.dim() == 4:
        return tensor.to(device, memory_format=torch.channels_last, **kwargs)
    return tensor.to(device, **kwargs)


def create_grad_scaler() -> torch.amp.GradScaler:
    enabled = USE_AMP and DEVICE.type == "cuda"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def autocast_context():
    enabled = USE_AMP and DEVICE.type == "cuda"
    return torch.autocast(device_type=DEVICE.type, enabled=enabled)
