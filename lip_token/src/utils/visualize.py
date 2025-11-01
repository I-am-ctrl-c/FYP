"""Visualisation helpers for reconstruction samples."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import torch

try:
    from torchvision.utils import make_grid, save_image
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "torchvision is required for visualisation utilities. "
        "Install it with `pip install torchvision`."
    ) from exc


def _denormalise(frame: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=frame.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=frame.device).view(-1, 1, 1)
    return frame * std_t + mean_t


def visualise_reconstruction(
    original: torch.Tensor,
    recon: torch.Tensor,
    *,
    save_path: Path,
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.5, 0.5, 0.5),
    max_samples: int = 4,
) -> None:
    """
    Save a grid comparing original vs reconstructed frames.

    Args:
        original: tensor (B, C, T, H, W)
        recon: tensor (B, C, T, H, W)
        save_path: destination image path
    """
    batch = min(original.shape[0], recon.shape[0], max_samples)
    if batch == 0:
        return

    frame_count = original.shape[2]
    frame_indices = [0, frame_count // 2, frame_count - 1] if frame_count > 2 else list(range(frame_count))
    images = []
    for idx in range(batch):
        for frame_idx in frame_indices:
            orig_frame = _denormalise(original[idx, :, frame_idx], mean, std).clamp(0, 1)
            recon_frame = _denormalise(recon[idx, :, frame_idx], mean, std).clamp(0, 1)
            images.extend([orig_frame, recon_frame])

    grid = make_grid(torch.stack(images, dim=0), nrow=len(frame_indices) * 2, padding=2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(save_path))


__all__ = ["visualise_reconstruction"]

