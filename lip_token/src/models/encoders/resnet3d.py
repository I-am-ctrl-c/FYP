"""3D ResNet encoder tailored for video VQ-VAE."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18


class ResNet3DEncoder(nn.Module):
    """ResNet3d encoder producing spatio-temporal feature maps."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        in_channels: int = 3,
        pretrained: bool = False,
        freeze_bn: bool = False,
    ) -> None:
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        base = r3d_18(weights=weights)

        if in_channels != 3:
            orig_conv: nn.Conv3d = base.stem[0]
            new_conv = nn.Conv3d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            if in_channels > 3:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            else:
                with torch.no_grad():
                    new_conv.weight[:, :in_channels] = orig_conv.weight[:, :in_channels]
            base.stem[0] = new_conv

        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.proj = nn.Conv3d(512, embedding_dim, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm3d(embedding_dim)
        self.activation = nn.ReLU(inplace=True)

        if freeze_bn:
            self._freeze_batchnorm()

    def _freeze_batchnorm(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()
                module.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shaped (B, C, T, H, W)
        Returns:
            torch.Tensor: latent tensor (B, embedding_dim, T', H', W')
        """
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.proj(out)
        out = self.norm(out)
        return self.activation(out)


def build_resnet3d_encoder(cfg: Mapping[str, object]) -> ResNet3DEncoder:
    """Factory to instantiate encoder from config mapping."""
    embedding_dim = int(cfg.get("embedding_dim", 256))
    in_channels = int(cfg.get("in_channels", 3))
    pretrained = bool(cfg.get("pretrained", False))
    freeze_bn = bool(cfg.get("freeze_bn", False))
    return ResNet3DEncoder(
        embedding_dim=embedding_dim,
        in_channels=in_channels,
        pretrained=pretrained,
        freeze_bn=freeze_bn,
    )


__all__ = ["ResNet3DEncoder", "build_resnet3d_encoder"]

