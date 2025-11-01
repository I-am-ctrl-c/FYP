"""Video transforms for VQ-VAE training."""

from __future__ import annotations

import random
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _temporal_sample(video: torch.Tensor, num_frames: int, jitter: bool = False) -> torch.Tensor:
    """Uniformly sample *num_frames* along time dimension."""
    total = video.shape[0]
    if total == num_frames:
        return video
    if total < num_frames:
        # Tile frames to reach target length
        repeat = num_frames // total + 1
        video = video.repeat(repeat, 1, 1, 1)
        total = video.shape[0]

    indices = torch.linspace(0, total - 1, steps=num_frames)
    if jitter:
        noise = torch.empty(num_frames).uniform_(-0.5, 0.5)
        indices = indices + noise
    indices = indices.clamp_(0, total - 1).round().long()
    return video.index_select(0, indices)


def _resize_video(video: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize spatial dimensions to *size* (H, W)."""
    if video.shape[-2:] == size:
        return video
    video_5d = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    resized = F.interpolate(
        video_5d,
        size=(video.shape[0], size[0], size[1]),
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0).permute(1, 0, 2, 3)


def _center_crop(video: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Center crop spatial dimensions."""
    h, w = video.shape[-2:]
    th, tw = size
    if (h, w) == (th, tw):
        return video
    i = (h - th) // 2
    j = (w - tw) // 2
    return video[..., i : i + th, j : j + tw]


def _random_crop(video: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Random crop spatial dimensions."""
    h, w = video.shape[-2:]
    th, tw = size
    if h == th and w == tw:
        return video
    if h < th or w < tw:
        return _center_crop(video, size)
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return video[..., i : i + th, j : j + tw]


def _maybe_horizontal_flip(video: torch.Tensor, prob: float) -> torch.Tensor:
    """Random horizontal flip."""
    if prob <= 0.0:
        return video
    if random.random() < prob:
        return torch.flip(video, dims=(-1,))
    return video


class VideoTransform:
    """Composable transform tailored for video tensors."""

    def __init__(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        mean: Sequence[float],
        std: Sequence[float],
        random_crop: bool = True,
        horizontal_flip: float = 0.0,
        temporal_jitter: bool = False,
    ) -> None:
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mean = torch.tensor(mean).view(-1, 1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1, 1)
        self.random_crop = random_crop
        self.horizontal_flip = horizontal_flip
        self.temporal_jitter = temporal_jitter

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply transform on *video* tensor shaped (T, C, H, W).

        Returns:
            torch.Tensor: transformed tensor of shape (C, T, H, W).
        """
        video = _temporal_sample(video, self.num_frames, jitter=self.temporal_jitter)
        video = _resize_video(video, (self.height, self.width))
        crop_size = (self.height, self.width)
        if self.random_crop:
            video = _random_crop(video, crop_size)
        else:
            video = _center_crop(video, crop_size)
        video = _maybe_horizontal_flip(video, self.horizontal_flip)

        # Normalize
        mean = self.mean.to(video.device)
        std = self.std.to(video.device)
        video = video.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
        video = (video - mean) / std
        return video


def build_video_transform(
    cfg: Mapping[str, object],
    *,
    train: bool,
) -> VideoTransform:
    """Factory to build :class:`VideoTransform` from config mapping."""
    num_frames = int(cfg.get("T") or cfg.get("num_frames") or 16)
    height = int(cfg.get("H") or cfg.get("height") or 112)
    width = int(cfg.get("W") or cfg.get("width") or 112)
    mean = cfg.get("mean", (0.5, 0.5, 0.5))
    std = cfg.get("std", (0.5, 0.5, 0.5))
    random_crop = bool(cfg.get("random_crop", True)) if train else False
    horizontal_flip = float(cfg.get("horizontal_flip", 0.5 if train else 0.0)) if train else 0.0
    temporal_jitter = bool(cfg.get("temporal_jitter", True)) if train else False
    return VideoTransform(
        num_frames=num_frames,
        height=height,
        width=width,
        mean=mean,  # type: ignore[arg-type]
        std=std,  # type: ignore[arg-type]
        random_crop=random_crop,
        horizontal_flip=horizontal_flip,
        temporal_jitter=temporal_jitter,
    )


__all__ = ["VideoTransform", "build_video_transform"]
