"""Utility functions for training and evaluation loops."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import torch
import yaml

LOG = logging.getLogger(__name__)


def deep_update(target: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge *src* into *target*."""
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            deep_update(target[key], value)  # type: ignore[index]
        else:
            target[key] = deepcopy(value)
    return target


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file into dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML file {path} must contain a mapping at the root.")
    return data


def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    """
    Load experiment config supporting simple defaults mechanism.

    Example format::

        defaults:
          - data: default
          - model: resnet3d_vqvae
          - train: default
        train:
          epochs: 10
    """
    config_root = config_path.parent.parent
    config = load_yaml(config_path)
    defaults = config.pop("defaults", [])
    merged: Dict[str, Any] = {}
    for item in defaults:
        if isinstance(item, str):
            section, name = item.split("/")
            path = config_root / section / f"{name}.yaml"
        elif isinstance(item, Mapping):
            if len(item) != 1:
                raise ValueError(f"Invalid defaults entry: {item}")
            section, name = next(iter(item.items()))
            path = config_root / section / f"{name}.yaml"
        else:
            raise TypeError(f"Unsupported defaults entry: {item}")
        section_cfg = load_yaml(path)
        deep_update(merged, {section: section_cfg})

    deep_update(merged, config)
    return merged


def prepare_device(device_str: str | None = None) -> torch.device:
    """Select a torch.device with helpful GPU reporting.

    Behavior:
    - If device_str provided and not 'auto', use it directly (e.g., 'cuda:2', 'cpu').
    - If no device_str or device_str == 'auto', prefer a CUDA device whose name contains '5090'.
      Otherwise, fall back to 'cuda:0' if available, else 'cpu'.
    Logs a summary of available CUDA devices and the chosen one.
    """
    # If explicitly specified (and not the special 'auto'), honour it.
    if device_str and device_str.lower() != "auto":
        device = torch.device(device_str)
        if device.type == "cuda" and torch.cuda.is_available():
            idx = device.index or 0
            try:
                LOG.info("CUDA device %d: %s", idx, torch.cuda.get_device_name(idx))
            except Exception:
                pass
        LOG.info("Using device %s", device)
        return device

    # Auto selection / default path
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        names = []
        for i in range(count):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = f"cuda:{i}"
            names.append(name)
            LOG.info("Detected CUDA device %d: %s", i, name)

        # Prefer any device whose reported name contains '5090'
        target_idx = None
        for i, name in enumerate(names):
            if "5090" in name.replace(" ", "").lower():
                target_idx = i
                break
        if target_idx is None:
            target_idx = 0
        device = torch.device(f"cuda:{target_idx}")
        LOG.info("Using device %s (%s)", device, names[target_idx])
        return device

    device = torch.device("cpu")
    LOG.info("Using device %s", device)
    return device


def save_checkpoint(state: Mapping[str, Any], path: Path) -> None:
    """Persist checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    LOG.info("Saved checkpoint to %s", path)


class MetricTracker:
    """Accumulate running averages of scalar metrics."""

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Mapping[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * n
            self.counts[key] = self.counts.get(key, 0) + n

    def average(self) -> Dict[str, float]:
        return {
            key: self.totals[key] / max(1, self.counts[key])
            for key in self.totals
        }


def detach_metrics(metrics: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    """Detach tensors into floating python numbers."""
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


__all__ = [
    "deep_update",
    "load_yaml",
    "load_experiment_config",
    "prepare_device",
    "save_checkpoint",
    "MetricTracker",
    "detach_metrics",
]
