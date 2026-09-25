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


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped by torch.compile."""
    return getattr(model, "_orig_mod", model)


def optimize_model(model: nn.Module) -> nn.Module:
    """Apply optional runtime optimizations that do not change model weights."""
    if USE_CHANNELS_LAST and DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if not TORCH_COMPILE or not hasattr(torch, "compile"):
        return model

    # Docker images often lack a C compiler; Triton/inductor then fails on first forward.
    # Suppress so training continues in eager mode instead of crashing.
    import torch._dynamo as dynamo

    dynamo.config.suppress_errors = True
    try:
        compiled = torch.compile(model)
        print("Using torch.compile (falls back to eager if inductor fails)")
        return compiled
    except Exception as exc:  # noqa: BLE001 - missing toolchain / unsupported backend
        print(f"torch.compile unavailable ({exc}); continuing without it")
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
