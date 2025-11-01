"""Command line entry point for training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

from ..engine.trainer_vqvae import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D-ResNet + VQ-VAE on lip videos.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp/m0_vqvae.yaml"),
        help="Path to experiment YAML configuration.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device string, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Override directory to store checkpoints and outputs.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    overrides: Dict[str, object] = {}
    if args.device:
        overrides["device"] = args.device
    if args.epochs is not None:
        overrides.setdefault("train", {})["epochs"] = args.epochs  # type: ignore[index]
    if args.save_dir is not None:
        overrides.setdefault("train", {})["save_dir"] = str(args.save_dir)  # type: ignore[index]
    run_training(args.config, overrides=overrides or None)


if __name__ == "__main__":
    main()

