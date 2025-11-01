"""Placeholder for future audio-visual fusion module (M5)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioVisualFusion(nn.Module):
    """Stub fusion module to combine video and audio representations."""

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:  # pragma: no cover - placeholder
        raise NotImplementedError("AudioVisualFusion will be implemented in M5.")


__all__ = ["AudioVisualFusion"]

