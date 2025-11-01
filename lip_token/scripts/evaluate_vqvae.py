"""Wrapper script for evaluating VQ-VAE checkpoints."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.evaluator_vqvae import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VQ-VAE checkpoint on a dataset split.")
    parser.add_argument("--config", type=Path, default=Path("configs/exp/m0_vqvae.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="valid_manifest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    metrics = evaluate_checkpoint(args.config, args.checkpoint, manifest_split=args.split)
    logging.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
