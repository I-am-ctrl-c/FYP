"""EMA Vector Quantizer implementation.

中文说明：
- 该模块将连续隐变量 (B, C, T, H, W) 量化为离散码本索引，并返回直通估计的量化张量。
- 为避免在 AMP(半精度) 下的数值溢出，距离计算与 EMA 更新统一在 float32 中进行。
- EMA 更新时，在除以簇大小时加入 epsilon，防止极小簇导致数值爆炸。
"""

from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizerEMA(nn.Module):
    """Vector quantiser with exponential moving average updates."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_beta: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding", embedding.clone())

    @torch.no_grad()
    def _update_embeddings(self, flat_inputs: torch.Tensor, encodings: torch.Tensor) -> None:
        cluster_sizes = encodings.sum(0)
        self.ema_cluster_size.mul_(self.decay).add_(cluster_sizes, alpha=1.0 - self.decay)

        embed_sum = encodings.t().matmul(flat_inputs)
        self.ema_embedding.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

        n = self.ema_cluster_size.sum()
        cluster_sizes = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )
        # 分母加 epsilon，避免极小簇放大造成数值溢出
        updated_embedding = self.ema_embedding / (cluster_sizes.unsqueeze(1) + self.epsilon)
        self.embedding.copy_(updated_embedding)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: latent tensor (B, C, T, H, W)
        Returns:
            quantized tensor, codebook loss, commitment loss, indices tensor (B, T, H, W)
        """
        b, c, t, h, w = inputs.shape
        # 为避免 AMP 下溢出，将距离与 EMA 更新固定在 float32
        inputs_perm = inputs.permute(0, 2, 3, 4, 1).contiguous()
        flat_inputs = inputs_perm.view(-1, c)
        flat_inputs_f32 = flat_inputs.float()
        embedding_f32 = self.embedding.float()

        distances = (
            flat_inputs_f32.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs_f32 @ embedding_f32.T
            + embedding_f32.pow(2).sum(dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_inputs_f32.dtype)

        quantized_f32 = encodings @ embedding_f32
        quantized_f32 = quantized_f32.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

        if self.training:
            self._update_embeddings(flat_inputs_f32, encodings)

        # 将量化结果转换回输入 dtype，用于直通估计和传播
        quantized = quantized_f32.to(dtype=inputs.dtype)
        quantized_stopped = inputs + (quantized - inputs).detach()

        # 损失在 float32 中计算更稳定
        codebook_loss = F.mse_loss(quantized_f32.detach(), inputs.float())
        commitment_loss = self.commitment_beta * F.mse_loss(inputs.float(), quantized_f32.detach())

        indices = encoding_indices.view(b, t, h, w)
        return quantized_stopped, codebook_loss, commitment_loss, indices


def build_quantizer(cfg: Mapping[str, object]) -> VectorQuantizerEMA:
    """Factory for EMA quantiser."""
    num_embeddings = int(cfg.get("codebook_size") or cfg.get("num_embeddings") or 512)
    embedding_dim = int(cfg.get("embedding_dim") or cfg.get("latent_dim") or 256)
    commitment_beta = float(cfg.get("commitment_beta", 0.25))
    decay = float(cfg.get("decay", 0.99))
    epsilon = float(cfg.get("epsilon", 1e-5))
    return VectorQuantizerEMA(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_beta=commitment_beta,
        decay=decay,
        epsilon=epsilon,
    )


__all__ = ["VectorQuantizerEMA", "build_quantizer"]
