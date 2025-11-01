"""Metrics utilities for reconstruction quality."""

from __future__ import annotations

import math

import torch


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return torch.mean((a - b) ** 2)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio."""
    mse_value = mse(a, b)
    if mse_value <= 1e-10:
        return torch.tensor(float("inf"), device=a.device)
    return 20 * torch.log10(torch.tensor(max_val, device=a.device)) - 10 * torch.log10(mse_value)


__all__ = ["mse", "psnr"]

