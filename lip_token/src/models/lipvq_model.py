"""Wrapper assembling encoder, quantiser, and decoder."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import build_resnet3d_encoder
from .vqvae import build_decoder, build_quantizer


class LipVQVAEModel(nn.Module):
    """End-to-end 3D ResNet + VQ-VAE model."""

    def __init__(
        self,
        encoder: nn.Module,
        quantizer: nn.Module,
        decoder: nn.Module,
        *,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        recon_loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        if recon_loss_type not in {"mse", "l1"}:
            raise ValueError("recon_loss_type must be 'mse' or 'l1'.")
        self.recon_loss_type = recon_loss_type

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        recon = self.decoder(z_q)
        if recon.shape[-3:] != target_shape[-3:]:
            recon = F.interpolate(
                recon, size=target_shape[-3:], mode="trilinear", align_corners=False
            )
        return recon

    def quantize(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantizer(z_e)

    def reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.recon_loss_type == "mse":
            return F.mse_loss(recon, target)
        return F.l1_loss(recon, target)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: input batch shaped (B, C, T, H, W)

        Returns:
            dict containing reconstructions, codebook indices, and losses.
        """
        z_e = self.encode(x)
        z_q, codebook_loss, commitment_loss, indices = self.quantize(z_e)
        recon = self.decode(z_q, target_shape=x.shape)
        recon_loss = self.reconstruction_loss(recon, x)
        loss = (
            recon_loss
            + self.codebook_weight * codebook_loss
            + self.commitment_weight * commitment_loss
        )
        return {
            "loss": loss,
            "loss_recon": recon_loss,
            "loss_codebook": codebook_loss,
            "loss_commit": commitment_loss,
            "z_e": z_e,
            "z_q": z_q,
            "recon": recon,
            "indices": indices,
        }


def build_lipvq_model(cfg: Mapping[str, object]) -> LipVQVAEModel:
    """Instantiate model from nested mapping config."""
    encoder_cfg = cfg.get("encoder", {})
    quantizer_cfg = cfg.get("quantizer", {})
    decoder_cfg = cfg.get("decoder", {})
    commitment_weight = float(cfg.get("commitment_beta", cfg.get("beta", 0.25)))
    codebook_weight = float(cfg.get("codebook_weight", cfg.get("gamma", 1.0)))
    recon_loss_type = str(cfg.get("recon_loss", "mse"))

    encoder = build_resnet3d_encoder(encoder_cfg)  # type: ignore[arg-type]
    quantizer = build_quantizer({**encoder_cfg, **quantizer_cfg})  # share embedding_dim
    decoder_input_cfg = {"embedding_dim": quantizer.embedding_dim}
    decoder = build_decoder({**decoder_input_cfg, **decoder_cfg})

    return LipVQVAEModel(
        encoder=encoder,
        quantizer=quantizer,
        decoder=decoder,
        commitment_weight=commitment_weight,
        codebook_weight=codebook_weight,
        recon_loss_type=recon_loss_type,
    )


__all__ = ["LipVQVAEModel", "build_lipvq_model"]

