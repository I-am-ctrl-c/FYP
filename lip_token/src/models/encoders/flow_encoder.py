"""Placeholder for future optical-flow encoder (M1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FlowEncoder(nn.Module):
    """Stub module reserved for flow-based features."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_not_implemented", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - placeholder
        raise NotImplementedError("FlowEncoder will be implemented in M1.")


__all__ = ["FlowEncoder"]

