"""Placeholder for optical flow consistency loss (M2)."""

from __future__ import annotations

import torch


def flow_consistency_loss(flow_rgb: torch.Tensor, flow_pred: torch.Tensor) -> torch.Tensor:  # pragma: no cover - placeholder
    raise NotImplementedError("Flow consistency loss will be implemented in M2.")


__all__ = ["flow_consistency_loss"]

