"""3D decoder mirroring encoder down-sampling pattern."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    """3D upsample block with interpolation followed by conv stack."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: Sequence[float]) -> None:
        super().__init__()
        self.scale_factor = tuple(float(s) for s in scale_factor)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="trilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        return self.conv(x)


class VQVAE3DDecoder(nn.Module):
    """Decoder reconstructing RGB clips from quantised latents."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dims: Sequence[int],
        output_channels: int = 3,
        upsample_scales: Sequence[Sequence[float]] = ((1, 2, 2), (1, 2, 2), (2, 2, 2)),
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != len(upsample_scales):
            raise ValueError("hidden_dims and upsample_scales must share the same length.")

        self.blocks = nn.ModuleList()
        in_channels = embedding_dim
        for out_channels, scale in zip(hidden_dims, upsample_scales):
            self.blocks.append(UpsampleBlock(in_channels, out_channels, scale))
            in_channels = out_channels

        self.head = nn.Conv3d(in_channels, output_channels, kernel_size=3, padding=1)
        self.final_activation = nn.Tanh() if final_activation else nn.Identity()

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: quantised latent tensor (B, C, T, H, W)
        Returns:
            torch.Tensor: reconstruction (B, 3, T*, H*, W*)
        """
        out = z_q
        for block in self.blocks:
            out = block(out)
        out = self.head(out)
        return self.final_activation(out)


def build_decoder(cfg: Mapping[str, object]) -> VQVAE3DDecoder:
    """Factory helper to create decoder from mapping config."""
    embedding_dim = int(cfg.get("embedding_dim") or cfg.get("latent_dim") or 256)
    hidden_dims = tuple(int(v) for v in cfg.get("hidden_dims", (256, 128, 64)))
    output_channels = int(cfg.get("output_channels", 3))
    upsample_scales = cfg.get("upsample_scales", ((1, 2, 2), (1, 2, 2), (2, 2, 2)))
    final_activation = bool(cfg.get("final_activation", False))
    return VQVAE3DDecoder(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        output_channels=output_channels,
        upsample_scales=upsample_scales,  # type: ignore[arg-type]
        final_activation=final_activation,
    )


__all__ = ["VQVAE3DDecoder", "build_decoder"]

