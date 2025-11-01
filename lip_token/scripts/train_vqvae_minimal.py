"""Minimal, single-file 3D-ResNet + VQ-VAE training script.

This script intentionally avoids project-wide modularisation for readability.
It supports the CMLR-style dataset where train/val/test CSVs contain one token
per line like: s9/20130719_section_2_000.71_002.14 and maps them to
video/<spk>/<spk>/<date>/section_*.mp4.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision.io import read_video as tv_read_video
    from torchvision.utils import save_image, make_grid
    from torchvision.models.video import R3D_18_Weights, r3d_18
except Exception:
    tv_read_video = None
    save_image = None
    make_grid = None
    r3d_18 = None
    R3D_18_Weights = None

LOG = logging.getLogger("vqvae_minimal")


# --------------- Utils ---------------

def seed_everything(seed: int = 42) -> None:
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def prepare_device(device: Optional[str] = None) -> torch.device:
    # Respect explicit selection
    if device and device.lower() != "auto":
        dev = torch.device(device)
        if dev.type == "cuda" and torch.cuda.is_available():
            idx = dev.index or 0
            try:
                LOG.info("CUDA device %d: %s", idx, torch.cuda.get_device_name(idx))
            except Exception:
                pass
        LOG.info("Using device %s", dev)
        return dev

    # Auto: prefer any GPU whose name includes '5090'
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
        target_idx = None
        for i, name in enumerate(names):
            if "5090" in name.replace(" ", "").lower():
                target_idx = i
                break
        if target_idx is None:
            target_idx = 0
        dev = torch.device(f"cuda:{target_idx}")
        LOG.info("Using device %s (%s)", dev, names[target_idx])
        return dev

    dev = torch.device("cpu")
    LOG.info("Using device %s", dev)
    return dev


def _guess_ext(base: Path) -> Path:
    if base.exists():
        return base
    for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
        p = Path(str(base) + ext)
        if p.exists():
            return p
    return Path(str(base) + ".mp4")


def token_to_video_path(token: str, video_root: Path) -> Path:
    token = token.strip().strip(",")
    if not token:
        raise ValueError("Empty token")
    if "/" not in token:
        return _guess_ext(video_root / token)
    spk, rest = token.split("/", 1)
    if "_" not in rest:
        return _guess_ext(video_root / spk / rest)
    date_part, remainder = rest.split("_", 1)
    filename = remainder if remainder.startswith("section_") else ("section_" + remainder)
    base = video_root / spk / spk / date_part / filename
    return _guess_ext(base)


def read_video_frames(path: Path) -> torch.Tensor:
    """Read a video into (T,C,H,W) in [0,1] using torchvision or decord if available."""
    # Prefer decord if installed
    try:
        import decord

        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(path))
        frames = vr.get_batch(list(range(len(vr))))  # (T,H,W,C) torch if bridge
        return frames.permute(0, 3, 1, 2).float() / 255.0
    except Exception:
        pass

    if tv_read_video is None:
        raise RuntimeError("torchvision is not available; install torchvision or decord.")
    video, _, _ = tv_read_video(str(path), output_format="TCHW")
    if video.numel() == 0:
        raise RuntimeError(f"No frames read from {path}")
    return video.float() / 255.0


def sample_temporal(video: torch.Tensor, T: int) -> torch.Tensor:
    total = video.shape[0]
    if total == T:
        return video
    if total < T:
        rep = T // total + 1
        video = video.repeat(rep, 1, 1, 1)
        total = video.shape[0]
    indices = torch.linspace(0, total - 1, steps=T).round().long()
    return video.index_select(0, indices)


def resize_video(video: torch.Tensor, H: int, W: int) -> torch.Tensor:
    if list(video.shape[-2:]) == [H, W]:
        return video
    x = video.permute(1, 0, 2, 3).unsqueeze(0)  # 1,C,T,H,W
    x = F.interpolate(x, size=(video.shape[0], H, W), mode="trilinear", align_corners=False)
    return x.squeeze(0).permute(1, 0, 2, 3)


def normalise(video: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    x = video.permute(1, 0, 2, 3)  # C,T,H,W
    mean_t = torch.tensor(mean, device=x.device).view(-1, 1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(-1, 1, 1, 1)
    return (x - mean_t) / std_t


# --------------- Dataset ---------------


@dataclass
class Sample:
    vid: str
    path: Path


class CMLRDataset(Dataset):
    def __init__(self, manifest: Path, T: int = 16, H: int = 112, W: int = 112) -> None:
        self.manifest = Path(manifest)
        if not self.manifest.exists():
            raise FileNotFoundError(self.manifest)
        self.T, self.H, self.W = T, H, W
        self.video_root = (self.manifest.parent / "video").resolve()

        tokens = [line.strip() for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        records: List[Sample] = []
        missing = 0
        for token in tokens:
            p = token_to_video_path(token, self.video_root)
            if p.exists():
                records.append(Sample(vid=token.replace("/", "_"), path=p))
            else:
                missing += 1
        if not records:
            raise RuntimeError("No existing videos found from manifest.")
        if missing:
            LOG.warning("Skipped %d missing videos from %s", missing, self.manifest)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        s = self.records[idx]
        frames = read_video_frames(s.path)  # (T,C,H,W)
        frames = sample_temporal(frames, self.T)
        frames = resize_video(frames, self.H, self.W)
        x = normalise(frames)  # (C,T,H,W)
        return {"rgb": x, "video_id": s.vid}


# --------------- Model ---------------


class ResNet3DEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        if r3d_18 is None:
            raise RuntimeError("torchvision video models not available; install torchvision.")
        base = r3d_18(weights=None)
        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.proj = nn.Conv3d(512, embedding_dim, kernel_size=1)
        self.norm = nn.BatchNorm3d(embedding_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25, decay: float = 0.99, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        emb = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", emb)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding", emb.clone())

    @torch.no_grad()
    def _update(self, flat_inputs: torch.Tensor, encodings: torch.Tensor) -> None:
        cluster_sizes = encodings.sum(0)
        self.ema_cluster_size.mul_(self.decay).add_(cluster_sizes, alpha=1.0 - self.decay)
        embed_sum = encodings.t().matmul(flat_inputs)
        self.ema_embedding.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)
        n = self.ema_cluster_size.sum()
        cluster_sizes = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
        self.embedding.copy_(self.ema_embedding / cluster_sizes.unsqueeze(1))

    def forward(self, x: torch.Tensor):
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, c)
        d = (y.pow(2).sum(1, keepdim=True) - 2 * y @ self.embedding.T + self.embedding.pow(2).sum(1))
        idx = torch.argmin(d, dim=1)
        enc = F.one_hot(idx, self.num_embeddings).type(y.dtype)
        z = enc @ self.embedding
        z = z.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        if self.training:
            self._update(y, enc)
        z_st = x + (z - x).detach()
        code_loss = F.mse_loss(z.detach(), x)
        commit_loss = self.beta * F.mse_loss(x, z.detach())
        return z_st, code_loss, commit_loss, idx.view(b, t, h, w)


class Decoder3D(nn.Module):
    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, 256, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LipVQVAE(nn.Module):
    def __init__(self, embedding_dim: int = 256, codebook_size: int = 512, beta: float = 0.25, gamma: float = 1.0) -> None:
        super().__init__()
        self.encoder = ResNet3DEncoder(embedding_dim=embedding_dim)
        self.quant = VectorQuantizerEMA(codebook_size, embedding_dim, beta)
        self.decoder = Decoder3D(embedding_dim=embedding_dim)
        self.beta = beta
        self.gamma = gamma

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, vq_loss, commit_loss, indices = self.quant(z_e)
        x_rec = self.decoder(z_q)
        return z_e, z_q, x_rec, vq_loss, commit_loss, indices


# --------------- Training ---------------


@dataclass
class TrainCfg:
    train_manifest: Path
    val_manifest: Optional[Path]
    save_dir: Path
    T: int = 16
    H: int = 112
    W: int = 112
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-2
    amp: bool = True
    embedding_dim: int = 256
    codebook_size: int = 512
    beta: float = 0.25
    gamma: float = 1.0
    seed: int = 42


def visualise_recon(orig: torch.Tensor, rec: torch.Tensor, path: Path) -> None:
    if save_image is None or make_grid is None:
        return
    b = min(2, orig.shape[0])
    t = orig.shape[2]
    idxs = [0, t // 2, t - 1] if t > 2 else list(range(t))
    imgs = []
    for i in range(b):
        for j in idxs:
            imgs.append(orig[i, :, j].clamp(0, 1))
            imgs.append(rec[i, :, j].clamp(0, 1))
    grid = make_grid(torch.stack(imgs), nrow=len(idxs) * 2, padding=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(path))


def train(cfg: TrainCfg, device: torch.device) -> None:
    seed_everything(cfg.seed)
    save_ckpt = cfg.save_dir / "checkpoints"
    save_rec = cfg.save_dir / "recon_samples"
    save_tok = cfg.save_dir / "tokens"
    save_ckpt.mkdir(parents=True, exist_ok=True)
    save_rec.mkdir(parents=True, exist_ok=True)
    save_tok.mkdir(parents=True, exist_ok=True)

    train_ds = CMLRDataset(cfg.train_manifest, T=cfg.T, H=cfg.H, W=cfg.W)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = None
    if cfg.val_manifest and Path(cfg.val_manifest).exists():
        val_ds = CMLRDataset(cfg.val_manifest, T=cfg.T, H=cfg.H, W=cfg.W)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=max(1, cfg.num_workers // 2))

    model = LipVQVAE(embedding_dim=cfg.embedding_dim, codebook_size=cfg.codebook_size, beta=cfg.beta, gamma=cfg.gamma).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {"loss": 0.0, "recon": 0.0, "vq": 0.0, "commit": 0.0}
        seen = 0
        for batch in train_loader:
            x = batch["rgb"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                z_e, z_q, x_rec, vq_loss, commit_loss, indices = model(x)
                # resize recon to match target size if needed
                if x_rec.shape[-3:] != x.shape[-3:]:
                    x_rec = F.interpolate(x_rec, size=x.shape[-3:], mode="trilinear", align_corners=False)
                recon_loss = F.mse_loss(x_rec, x)
                loss = recon_loss + vq_loss + commit_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bsz = x.size(0)
            seen += bsz
            running["loss"] += float(loss.detach().cpu()) * bsz
            running["recon"] += float(recon_loss.detach().cpu()) * bsz
            running["vq"] += float(vq_loss.detach().cpu()) * bsz
            running["commit"] += float(commit_loss.detach().cpu()) * bsz

        avg = {k: v / max(1, seen) for k, v in running.items()}
        LOG.info("Epoch %d: loss=%.4f recon=%.4f vq=%.4f commit=%.4f", epoch, avg["loss"], avg["recon"], avg["vq"], avg["commit"])

        # one quick visual from train set
        with torch.no_grad():
            x = next(iter(train_loader))["rgb"].to(device)
            _, _, x_rec, _, _, _ = model(x)
            if x_rec.shape[-3:] != x.shape[-3:]:
                x_rec = F.interpolate(x_rec, size=x.shape[-3:], mode="trilinear", align_corners=False)
            visualise_recon(x[:2].cpu(), x_rec[:2].cpu(), save_rec / f"epoch_{epoch:03d}.png")

        torch.save(model.encoder.state_dict(), save_ckpt / "encoder.pt")
        torch.save(model.quant.state_dict(), save_ckpt / "quantizer.pt")
        torch.save(model.decoder.state_dict(), save_ckpt / "decoder.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal VQ-VAE training on CMLR data")
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, default=None)
    p.add_argument("--save-dir", type=Path, default=Path("./outputs/m0_vqvae_minimal"))
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--H", type=int, default=112)
    p.add_argument("--W", type=int, default=112)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--codebook-size", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    device = prepare_device(args.device)
    cfg = TrainCfg(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        save_dir=args.save_dir,
        T=args.T,
        H=args.H,
        W=args.W,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        codebook_size=args.codebook_size,
        beta=args.beta,
        gamma=args.gamma,
        seed=args.seed,
    )
    train(cfg, device)


if __name__ == "__main__":
    main()
