"""Placeholder for future CTC/Transformer head (M4)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LipCTCHead(nn.Module):
    """Stub classification head for lip-reading tokens."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover - placeholder
        raise NotImplementedError("LipCTCHead will be implemented in M4.")


__all__ = ["LipCTCHead"]

