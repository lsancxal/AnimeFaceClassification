"""Reproducibility helpers."""

from __future__ import annotations

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python-facing RNGs used by NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
