"""Training loop implementation for lip VQ-VAE."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore

try:
    import torchvision.utils as vutils  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    vutils = None  # type: ignore

from ..datamodules import LipVQVaeDataset, build_video_transform
from ..models import LipVQVAEModel, build_lipvq_model
from ..utils import seed_everything, visualise_reconstruction
from .utils import (
    MetricTracker,
    deep_update,
    detach_metrics,
    load_experiment_config,
    prepare_device,
    save_checkpoint,
)

LOG = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration harvested from YAML."""

    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: Optional[float] = 1.0
    amp: bool = True
    amp_start_epoch: int = 3  # 从该轮起启用 AMP（逐步恢复稳定性）
    save_dir: Path = Path("./outputs/m0_vqvae")
    log_interval: int = 200
    val_interval: int = 1
    seed: int = 42
    resume: Optional[Path] = None
    num_devices: int = 1
    scheduler: Optional[Mapping[str, object]] = None
    tensorboard: bool = True
    tb_log_dir: Optional[Path] = None
    checkpoint_interval: int = 10  # 额外保存 epoch 的间隔

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "TrainerConfig":
        kwargs: Dict[str, object] = {}
        for field in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            if field in mapping:
                value = mapping[field]
                if field in {"save_dir", "resume", "tb_log_dir"} and value:
                    kwargs[field] = Path(value)
                else:
                    kwargs[field] = value
        return cls(**kwargs)


def create_dataloaders(cfg: Mapping[str, object]) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Instantiate train/validation dataloaders based on config."""
    train_manifest = cfg.get("train_manifest") or cfg.get("manifest")
    if not train_manifest:
        raise ValueError("Data config must specify 'manifest' or 'train_manifest'.")
    val_manifest = cfg.get("valid_manifest") or cfg.get("val_manifest")

    backend = cfg.get("backend", "decord")
    num_frames = int(cfg.get("T", cfg.get("num_frames", 16)))
    stride = int(cfg.get("stride", 1))
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 4))
    root_dir = cfg.get("root_dir")
    video_column = cfg.get("video_column", "video_path")
    id_column = cfg.get("id_column", "utt_id")

    transform_train = build_video_transform(cfg, train=True)
    transform_val = build_video_transform(cfg, train=False)

    train_dataset = LipVQVaeDataset(
        manifest=train_manifest,
        backend=backend,
        transform=transform_train,
        num_frames=num_frames,
        stride=stride,
        root_dir=root_dir,
        video_column=video_column,
        id_column=id_column,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader: Optional[DataLoader] = None
    if val_manifest:
        val_dataset = LipVQVaeDataset(
            manifest=val_manifest,
            backend=backend,
            transform=transform_val,
            num_frames=num_frames,
            stride=stride,
            root_dir=root_dir,
            video_column=video_column,
            id_column=id_column,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=True,
        )

    return train_loader, val_loader


def _build_optimizer(model: torch.nn.Module, cfg: TrainerConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainerConfig):
    if not cfg.scheduler:
        return None
    name = str(cfg.scheduler.get("name", "cosine")).lower()
    params = cfg.scheduler.get("params", {})
    if name == "cosine":
        eta_min = float(params.get("eta_min", cfg.lr * 0.1))
        t_max = int(params.get("t_max", 10))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if name == "step":
        step_size = int(params.get("step_size", 10))
        gamma = float(params.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "piecewise":
        class PiecewiseConstantLR:
            """Piecewise constant learning-rate schedule."""

            def __init__(self, opt: torch.optim.Optimizer, milestones, lrs) -> None:
                self.optimizer = opt
                self.milestones = [int(m) for m in milestones]
                self.lrs = [float(lr) for lr in lrs]
                if len(self.lrs) != len(self.milestones) + 1:
                    raise ValueError("piecewise scheduler requires len(lrs) == len(milestones) + 1")

            def get_lr_for_epoch(self, epoch: int) -> float:
                idx = 0
                for milestone in self.milestones:
                    if epoch >= milestone:
                        idx += 1
                idx = min(idx, len(self.lrs) - 1)
                return self.lrs[idx]

            def step(self, epoch: Optional[int] = None) -> None:
                if epoch is None:
                    return
                lr = self.get_lr_for_epoch(int(epoch))
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

        return PiecewiseConstantLR(
            optimizer,
            params.get("milestones", []),
            params.get("lrs", [cfg.lr]),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


class VQVAETrainer:
    """Manage training/validation of VQ-VAE model."""

    def __init__(
        self,
        model: LipVQVAEModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        trainer_cfg: TrainerConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = trainer_cfg
        self.device = device
        self.save_dir = trainer_cfg.save_dir
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.recon_dir = self.save_dir / "recon_samples"
        self.token_dir = self.save_dir / "tokens"
        self.optimizer = _build_optimizer(self.model, trainer_cfg)
        self.scheduler = _build_scheduler(self.optimizer, trainer_cfg)
        self.global_step = 0
        self.best_val = float("inf")
        self.model.train()

        self.writer: Optional[SummaryWriter] = None
        if trainer_cfg.tensorboard and SummaryWriter is not None:
            tb_dir = trainer_cfg.tb_log_dir or (self.save_dir / "tb")
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir))

    def _to_device(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }  # type: ignore[dict-item]

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        tracker = MetricTracker()
        self.model.train()
        for step, batch in enumerate(self.train_loader, start=1):
            self.global_step += 1
            batch = self._to_device(batch)
            inputs = batch["rgb"]

            if epoch == 1 and step == 1:
                LOG.info(
                    "Input shape (B,C,T,H,W) will be: (%d,%d,%d,%d,%d)",
                    inputs.size(0),
                    inputs.size(1),
                    inputs.size(2),
                    inputs.size(3),
                    inputs.size(4),
                )

            self.optimizer.zero_grad(set_to_none=True)

            amp_now = bool(
                self.cfg.amp
                and self.device.type == "cuda"
                and epoch >= self.cfg.amp_start_epoch
            )
            with torch.amp.autocast("cuda", enabled=amp_now):
                outputs = self.model(inputs)
                loss = outputs["loss"]

            if not torch.isfinite(loss):
                LOG.warning("Non-finite loss detected; skipping optimisation step.")
                continue

            if amp_now:
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            scalar_keys = ["loss", "loss_recon", "loss_codebook", "loss_commit"]
            scalar_metrics = {key: outputs[key] for key in scalar_keys if key in outputs}
            tracker.update(detach_metrics(scalar_metrics), inputs.size(0))

            if "indices" in outputs:
                with torch.no_grad():
                    idx = outputs["indices"].reshape(-1).to("cpu")
                    K = getattr(self.model.quantizer, "num_embeddings", None)
                    if K is not None:
                        hist = torch.bincount(idx, minlength=int(K)).float()
                        prob = hist / max(1.0, float(hist.sum()))
                        perplexity = torch.exp(-(prob * torch.log(prob + 1e-10)).sum()).item()
                        used = float((hist > 0).sum().item())
                        tracker.update({"perplexity": perplexity, "codes_used": used}, n=1)

            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            if step % self.cfg.log_interval == 0 or step == len(self.train_loader):
                averages = tracker.average()
                log_msg = (
                    "[Epoch %d | Step %d] loss=%.4f recon=%.4f codebook=%.4f commit=%.4f"
                    % (
                        epoch,
                        step,
                        averages.get("loss", 0.0),
                        averages.get("loss_recon", 0.0),
                        averages.get("loss_codebook", 0.0),
                        averages.get("loss_commit", 0.0),
                    )
                )
                if "perplexity" in averages:
                    log_msg += " | ppl=%.2f used=%d" % (
                        averages["perplexity"],
                        int(averages.get("codes_used", 0.0)),
                    )
                LOG.info(log_msg)

                if self.writer is not None:
                    lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
                    self.writer.add_scalar("train/loss", averages.get("loss", 0.0), self.global_step)
                    self.writer.add_scalar("train/recon", averages.get("loss_recon", 0.0), self.global_step)
                    self.writer.add_scalar("train/codebook", averages.get("loss_codebook", 0.0), self.global_step)
                    self.writer.add_scalar("train/commit", averages.get("loss_commit", 0.0), self.global_step)
                    if "perplexity" in averages:
                        self.writer.add_scalar("train/perplexity", averages["perplexity"], self.global_step)
                        self.writer.add_scalar("train/codes_used", averages.get("codes_used", 0.0), self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    self.writer.flush()

        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if hasattr(self.scheduler, "step"):
                try:
                    self.scheduler.step(epoch)
                except TypeError:
                    self.scheduler.step()

        return tracker.average()

    @torch.no_grad()
    def validate(self, epoch: int) -> Optional[Dict[str, float]]:
        if not self.val_loader:
            return None
        self.model.eval()
        tracker = MetricTracker()
        sample_logged = False
        for batch in self.val_loader:
            batch = self._to_device(batch)
            inputs = batch["rgb"]
            outputs = self.model(inputs)
            scalar_keys = ["loss", "loss_recon", "loss_codebook", "loss_commit"]
            scalar_metrics = {key: outputs[key] for key in scalar_keys if key in outputs}
            tracker.update(detach_metrics(scalar_metrics), inputs.size(0))

            if not sample_logged:
                save_path = self.recon_dir / f"epoch_{epoch:03d}.png"
                visualise_reconstruction(
                    inputs.detach().cpu(),
                    outputs["recon"].detach().cpu(),
                    save_path=save_path,
                )
                if self.writer is not None and vutils is not None:
                    frames = []
                    t = inputs.size(2)
                    frame_ids = [0, t // 2, t - 1] if t > 2 else list(range(t))
                    for i in range(min(inputs.size(0), 2)):
                        for fid in frame_ids:
                            frames.append(inputs[i, :, fid].cpu().clamp(0, 1))
                            frames.append(outputs["recon"][i, :, fid].cpu().clamp(0, 1))
                    grid = vutils.make_grid(torch.stack(frames, dim=0), nrow=len(frame_ids) * 2)
                    self.writer.add_image("val/reconstruction_grid", grid, global_step=epoch)
                sample_logged = True

        averages = tracker.average()
        LOG.info(
            "[Validation | Epoch %d] loss=%.4f recon=%.4f codebook=%.4f commit=%.4f",
            epoch,
            averages.get("loss", 0.0),
            averages.get("loss_recon", 0.0),
            averages.get("loss_codebook", 0.0),
            averages.get("loss_commit", 0.0),
        )
        if self.writer is not None:
            self.writer.add_scalar("val/loss", averages.get("loss", 0.0), epoch)
            self.writer.add_scalar("val/recon", averages.get("loss_recon", 0.0), epoch)
            self.writer.add_scalar("val/codebook", averages.get("loss_codebook", 0.0), epoch)
            self.writer.add_scalar("val/commit", averages.get("loss_commit", 0.0), epoch)
            self.writer.flush()
        return averages

    def save_checkpoint(self, epoch: int, metrics: Mapping[str, float]) -> None:
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": dict(metrics),
        }
        save_checkpoint(state, self.checkpoint_dir / f"epoch_{epoch:03d}.pt")

    def save_components(self) -> None:
        checkpoint_path = self.save_dir / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.encoder.state_dict(), checkpoint_path / "encoder.pt")
        torch.save(self.model.quantizer.state_dict(), checkpoint_path / "quantizer.pt")
        torch.save(self.model.decoder.state_dict(), checkpoint_path / "decoder.pt")
        LOG.info("Exported encoder/quantizer/decoder weights to %s", checkpoint_path)

    def _save_periodic_checkpoint(self, epoch: int, metrics: Mapping[str, float]) -> None:
        if epoch % max(1, self.cfg.checkpoint_interval) == 0:
            path = self.checkpoint_dir / f"epoch_{epoch:03d}_snapshot.pt"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "metrics": dict(metrics),
                },
                path,
            )
            LOG.info("Saved periodic checkpoint at epoch %d to %s", epoch, path)

    def fit(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.recon_dir.mkdir(parents=True, exist_ok=True)
        self.token_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            LOG.info("Epoch %d training metrics: %s", epoch, train_metrics)
            if self.writer is not None:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f"epoch/train_{key}", value, epoch)
                self.writer.flush()

            val_metrics = None
            if self.val_loader and (epoch % self.cfg.val_interval == 0):
                val_metrics = self.validate(epoch)
                if val_metrics and val_metrics["loss"] < self.best_val:
                    self.best_val = val_metrics["loss"]
                    self.save_checkpoint(epoch, val_metrics)
                    LOG.info("New best validation loss %.4f at epoch %d", self.best_val, epoch)
                    self.checkpoint_dir.joinpath("best.pt").write_bytes(
                        (self.checkpoint_dir / f"epoch_{epoch:03d}.pt").read_bytes()
                    )
            else:
                self.save_checkpoint(epoch, train_metrics)

            self._save_periodic_checkpoint(epoch, val_metrics or train_metrics)

        self.save_components()

        if self.writer is not None:
            self.writer.close()


def run_training(exp_config_path: Path, overrides: Optional[Mapping[str, object]] = None) -> None:
    """High-level entry to launch training from config file."""
    cfg = load_experiment_config(exp_config_path)
    if overrides:
        deep_update(cfg, overrides)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = TrainerConfig.from_mapping(cfg.get("train", {}))
    seed_everything(train_cfg.seed)
    device = prepare_device(str(cfg.get("device")) if cfg.get("device") else None)

    model = build_lipvq_model(model_cfg)
    train_loader, val_loader = create_dataloaders(data_cfg)
    trainer = VQVAETrainer(model, train_loader, val_loader, train_cfg, device)
    trainer.fit()


__all__ = [
    "TrainerConfig",
    "VQVAETrainer",
    "create_dataloaders",
    "run_training",
]
