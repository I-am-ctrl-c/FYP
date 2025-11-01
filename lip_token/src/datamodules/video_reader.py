"""Video reading utilities supporting multiple backends."""

from __future__ import annotations

import enum
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

LOG = logging.getLogger(__name__)


class VideoReaderBackend(str, enum.Enum):
    """Enumeration of supported video decoding backends."""

    DECORD = "decord"
    PYAV = "pyav"
    TORCHVISION = "torchvision"

    @classmethod
    def from_string(cls, name: str) -> "VideoReaderBackend":
        """Return enum value matching *name*, case insensitive."""
        if isinstance(name, cls):
            return name
        try:
            return cls(str(name).lower())
        except ValueError as exc:
            valid = ", ".join(member.value for member in cls)
            raise ValueError(f"Unsupported video backend '{name}'. Expected one of: {valid}") from exc


def _sample_indices(length: int, num_samples: Optional[int], stride: int = 1) -> np.ndarray:
    """Return frame indices for uniform sampling."""
    if length <= 0:
        raise ValueError("Video contains no frames.")

    effective_len = max(1, length // stride)
    if num_samples is None or num_samples >= effective_len:
        indices = np.arange(0, length, stride, dtype=np.int64)
    else:
        collapsed = np.linspace(0, effective_len - 1, num_samples, dtype=np.float32)
        indices = (collapsed * stride).round().astype(np.int64)
    return np.clip(indices, 0, length - 1)


def _load_with_decord(path: Path, indices: Sequence[int]) -> torch.Tensor:
    try:
        import decord  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "decord is not installed. Install it with `pip install decord` or "
            "choose a different video backend."
        ) from exc

    decord.bridge.set_bridge("torch")
    reader = decord.VideoReader(str(path))
    frames = reader.get_batch(indices)  # shape: (num_indices, H, W, C)
    return frames.permute(0, 3, 1, 2).float() / 255.0


def _load_all_frames_with_pyav(path: Path) -> torch.Tensor:
    try:
        import av  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyAV is not installed. Install it with `pip install av` or "
            "use a different video backend."
        ) from exc

    container = av.open(str(path))
    frames = []
    for frame in container.decode(video=0):
        image = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(image).permute(2, 0, 1).float() / 255.0)

    if not frames:
        raise RuntimeError(f"No frames decoded from {path}.")
    return torch.stack(frames, dim=0)


def _load_all_frames_with_torchvision(path: Path) -> torch.Tensor:
    try:
        from torchvision.io import read_video  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for torchvision video backend. "
            "Install it with `pip install torchvision`."
        ) from exc

    video, _, _ = read_video(str(path), output_format="TCHW")
    if video.numel() == 0:
        raise RuntimeError(f"No frames read from {path}.")
    return video.float() / 255.0


def read_video_frames(
    path: Path,
    *,
    backend: VideoReaderBackend = VideoReaderBackend.DECORD,
    num_frames: Optional[int] = None,
    stride: int = 1,
) -> torch.Tensor:
    """Read *path* into a tensor with shape (T, C, H, W) and float32 values."""
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    backend = VideoReaderBackend.from_string(backend)

    if backend is VideoReaderBackend.DECORD:
        try:
            import decord  # type: ignore
        except ImportError:
            LOG.warning("Decord backend unavailable. Falling back to torchvision.")
            return read_video_frames(
                path, backend=VideoReaderBackend.TORCHVISION, num_frames=num_frames, stride=stride
            )
        reader = decord.VideoReader(str(path))
        total_frames = len(reader)
        indices = _sample_indices(total_frames, num_frames, stride=stride)
        frames = _load_with_decord(path, indices)
    elif backend is VideoReaderBackend.PYAV:
        frames = _load_all_frames_with_pyav(path)
    else:
        frames = _load_all_frames_with_torchvision(path)

    if backend is not VideoReaderBackend.DECORD:
        total_frames = frames.shape[0]
        indices = _sample_indices(total_frames, num_frames, stride=stride)
        frames = frames[indices]

    return frames


__all__ = ["VideoReaderBackend", "read_video_frames"]
