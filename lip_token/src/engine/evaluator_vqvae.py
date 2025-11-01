"""Utilities for exporting quantised tokens from a trained model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datamodules import LipVQVaeDataset, build_video_transform
from ..models import LipVQVAEModel, build_lipvq_model
from ..utils import seed_everything, visualise_reconstruction
from .utils import load_experiment_config, prepare_device, deep_update

LOG = logging.getLogger(__name__)


class TokenExporter:
    """Generate discrete token sequences and optional reconstructions."""

    def __init__(
        self,
        model: LipVQVAEModel,
        dataloader: DataLoader,
        device: torch.device,
        *,
        output_dir: Path,
        save_recon: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.output_dir = output_dir
        self.tokens_dir = output_dir / "tokens"
        self.recon_dir = output_dir / "recon_samples"
        self.save_recon = save_recon
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        if save_recon:
            self.recon_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def export(self) -> None:
        self.model.eval()
        for batch in self.dataloader:
            video_ids = batch["video_id"]
            inputs = batch["rgb"].to(self.device)

            outputs = self.model(inputs)
            indices = outputs["indices"].detach().cpu().numpy()
            for idx, video_id in enumerate(video_ids):
                dest = self.tokens_dir / f"{video_id}.npy"
                np.save(dest, indices[idx])
                LOG.info("Saved tokens to %s", dest)

            if self.save_recon:
                for idx, video_id in enumerate(video_ids):
                    save_path = self.recon_dir / f"{video_id}.png"
                    visualise_reconstruction(
                        inputs[idx : idx + 1].detach().cpu(),
                        outputs["recon"][idx : idx + 1].detach().cpu(),
                        save_path=save_path,
                    )


def _build_dataloader(cfg: Mapping[str, object], manifest_key: str, batch_size: int = 4) -> DataLoader:
    manifest = cfg.get(manifest_key)
    if not manifest:
        raise ValueError(f"Data config must include '{manifest_key}' for token export.")
    transform = build_video_transform(cfg, train=False)
    dataset = LipVQVaeDataset(
        manifest=manifest,
        backend=cfg.get("backend", "decord"),
        transform=transform,
        num_frames=int(cfg.get("T", 16)),
        stride=int(cfg.get("stride", 1)),
        root_dir=cfg.get("root_dir"),
        video_column=cfg.get("video_column", "video_path"),
        id_column=cfg.get("id_column", "utt_id"),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(cfg.get("num_workers", 4)))


def run_export(
    exp_config_path: Path,
    checkpoint_path: Path,
    *,
    manifest_split: str = "test_manifest",
    output_dir: Optional[Path] = None,
    save_recon: bool = False,
) -> None:
    """High-level helper for token export via CLI."""
    cfg = load_experiment_config(exp_config_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    seed_everything(int(train_cfg.get("seed", 42)))
    device = prepare_device(str(cfg.get("device")) if cfg.get("device") else None)

    model = build_lipvq_model(model_cfg)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    LOG.info("Loaded checkpoint %s", checkpoint_path)

    dataloader = _build_dataloader(data_cfg, manifest_split, batch_size=int(data_cfg.get("batch_size", 4)))
    output_root = output_dir or Path(train_cfg.get("save_dir", "./outputs/m0_vqvae"))
    exporter = TokenExporter(model, dataloader, device, output_dir=output_root, save_recon=save_recon)
    exporter.export()


@torch.no_grad()
def evaluate_checkpoint(
    exp_config_path: Path,
    checkpoint_path: Path,
    *,
    manifest_split: str = "valid_manifest",
    overrides: Optional[Mapping[str, object]] = None,
) -> Mapping[str, float]:
    """Evaluate reconstruction and quantisation losses for a checkpoint."""
    cfg = load_experiment_config(exp_config_path)
    if overrides:
        deep_update(cfg, overrides)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    seed_everything(int(train_cfg.get("seed", 42)))
    device = prepare_device(str(cfg.get("device")) if cfg.get("device") else None)

    dataloader = _build_dataloader(data_cfg, manifest_split, batch_size=int(data_cfg.get("batch_size", 4)))

    model = build_lipvq_model(model_cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    totals: Dict[str, float] = {"loss": 0.0, "loss_recon": 0.0, "loss_codebook": 0.0, "loss_commit": 0.0}
    count = 0
    for batch in dataloader:
        inputs = batch["rgb"].to(device)
        outputs = model(inputs)
        batch_size = inputs.size(0)
        count += batch_size
        for key in totals:
            totals[key] += float(outputs[key].detach().cpu()) * batch_size

    return {key: value / max(1, count) for key, value in totals.items()}


__all__ = ["TokenExporter", "run_export"]
