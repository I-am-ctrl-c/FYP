"""CLI for exporting discrete lip tokens from trained model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..engine.evaluator_vqvae import run_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VQ-VAE codebook indices for lip videos.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp/m0_vqvae.yaml"),
        help="Experiment configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_manifest",
        help="Which manifest key to use (e.g. 'test_manifest' or 'valid_manifest').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory where tokens will be stored. Defaults to training save_dir.",
    )
    parser.add_argument(
        "--save-recon",
        action="store_true",
        help="Whether to also save reconstruction visualisations.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run_export(
        args.config,
        args.checkpoint,
        manifest_split=args.split,
        output_dir=args.output,
        save_recon=args.save_recon,
    )


if __name__ == "__main__":
    main()

