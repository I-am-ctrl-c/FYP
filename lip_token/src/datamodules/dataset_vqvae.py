"""Dataset for lip VQ-VAE training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from .video_reader import VideoReaderBackend, read_video_frames


def _guess_ext(base: Path) -> Path:
    """Return an existing path by trying common video extensions.

    Note: do NOT use Path.with_suffix here because tokens often include
    floating timestamps like '..._010.31_012.98' which would be misread
    as a suffix and truncated. Build candidates by string concatenation.
    """
    if base.exists():
        return base
    for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
        p = Path(str(base) + ext)
        if p.exists():
            return p
    # fallback to .mp4 by convention
    return Path(str(base) + ".mp4")


def _cmlr_token_to_path(token: str, video_root: Path) -> Path:
    """Map CMLR token like 's9/20130719_section_2_000.71_002.14' to real path.

    Expected structure: video/<spk>/<spk>/<date>/section_*.mp4
    """
    token = token.strip().strip(",")
    if not token:
        raise ValueError("Empty token in manifest.")
    if "/" not in token:
        # fallback: treat token as relative path under video_root
        base = video_root / token
        return _guess_ext(base)
    spk, rest = token.split("/", 1)
    if "_" not in rest:
        base = video_root / spk / rest
        return _guess_ext(base)
    date_part, remainder = rest.split("_", 1)
    filename = remainder if remainder.startswith("section_") else ("section_" + remainder)

    # Try common layouts: single spk folder first (preferred), then double spk.
    candidates = [
        video_root / spk / date_part / filename,            # .../video/s7/20141017/section_*.mp4
        video_root / spk / spk / date_part / filename,      # .../video/s7/s7/20141017/section_*.mp4
    ]

    for base in candidates:
        p = _guess_ext(base)
        if p.exists():
            return p

    # Fallback: recursive search under speaker directory
    try:
        # Search limited depth under the speaker directory to avoid heavy globbing
        speaker_dir = video_root / spk
        pattern = filename + "*"
        for found in speaker_dir.rglob(filename + "*"):
            if found.is_file():
                return found
    except Exception:
        pass

    # Return the first candidate with .mp4 suffix as a final fallback
    return _guess_ext(candidates[0])


@dataclass
class SampleRecord:
    """Metadata for a single video sample."""

    video_id: str
    video_path: Path
    metadata: Mapping[str, str]


class LipVQVaeDataset(Dataset):
    """Iterable dataset returning videos as tensors ready for VQ-VAE."""

    def __init__(
        self,
        manifest: Union[str, Path],
        *,
        backend: Union[str, VideoReaderBackend] = VideoReaderBackend.DECORD,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_frames: Optional[int] = None,
        stride: int = 1,
        root_dir: Optional[Union[str, Path]] = None,
        video_column: str = "video_path",
        id_column: str = "utt_id",
    ) -> None:
        self.manifest_path = Path(manifest)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.backend = VideoReaderBackend.from_string(backend)
        self.transform = transform
        self.num_frames = num_frames
        self.stride = stride
        self.video_column = video_column
        self.id_column = id_column
        self.root_dir = Path(root_dir) if root_dir is not None else self.manifest_path.parent

        self.records: List[SampleRecord] = self._parse_manifest()

    def _parse_manifest(self) -> List[SampleRecord]:
        records: List[SampleRecord] = []
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            peek = handle.readline()
            handle.seek(0)
            # Heuristic: headered CSV if the first line contains a comma and alphabetic header tokens
            headered = "," in peek
            if headered:
                reader = csv.DictReader(handle)
                if self.video_column not in (reader.fieldnames or []):
                    # Fallback to CMLR format even if header present but doesn't include video_column
                    headered = False
                else:
                    for row in reader:
                        raw_path = row[self.video_column]
                        video_path = Path(raw_path)
                        if not video_path.is_absolute():
                            video_path = (self.root_dir / video_path).resolve()
                        video_id = row.get(self.id_column) or video_path.stem
                        records.append(SampleRecord(video_id=video_id, video_path=video_path, metadata=row))

            if not headered:
                # CMLR simple list: one token per line -> derive path under video/ tree
                reader_simple = csv.reader(handle)
                video_root = (self.root_dir / "video").resolve()
                for row in reader_simple:
                    if not row:
                        continue
                    token = row[0]
                    video_path = _cmlr_token_to_path(token, video_root)
                    video_id = token.replace("/", "_")
                    records.append(SampleRecord(video_id=video_id, video_path=video_path, metadata={"token": token}))
        if not records:
            raise RuntimeError(f"Manifest {self.manifest_path} contains no samples.")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        frames = read_video_frames(
            record.video_path,
            backend=self.backend,
            num_frames=self.num_frames,
            stride=self.stride,
        )
        # frames: (T, C, H, W)
        # 训练模型期望输入为 (B, C, T, H, W)，因此这里将单条样本变换为 (C, T, H, W)
        # 并在 DataLoader 维度上由 collate 叠加 batch。
        if self.transform:
            tensor = self.transform(frames)
        else:
            tensor = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        # 注意：不要返回原始 frames（各样本的空间尺寸可能不同，默认 collate 会尝试 stack 而失败）
        return {
            "rgb": tensor,  # (C, T, H, W)
            "video_id": record.video_id,
            "path": str(record.video_path),
            "meta": record.metadata,
        }


__all__ = ["LipVQVaeDataset", "SampleRecord"]
