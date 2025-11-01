# Lip Token VQ-VAE

Lip Token is a self-supervised framework that learns discrete representations of lip motion from RGB videos. It combines a 3D ResNet encoder with a VQ-VAE bottleneck to quantise video clips into token sequences that can serve downstream lip-reading and multi-modal tasks.

## Environment Setup

The project targets Python 3.10+ with PyTorch ≥ 2.2.

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux / macOS
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA/CPU
pip install pyyaml numpy decord av tqdm tensorboard
```

Optional packages:

- `decord` (recommended) for high-performance video decoding.
- `av` (PyAV) as a fallback decoder.

## Data Preparation

Place the preprocessed CMLR (news broadcast) dataset under `lip_token/data`. The expected manifest files follow:

```
lip_token/
└─ data/
   ├─ train.csv
   ├─ valid.csv
   ├─ test.csv
   ├─ videos/<video clips>.mp4
   ├─ audios/<audio clips>.wav         # optional for future work
   └─ texts/<transcripts>.txt          # optional for future work
```

Each CSV row should contain at least:

```
utt_id,video_path
0001,./videos/0001.mp4
```

Paths can be absolute or relative to the CSV file location.

## Training

```bash
cd lip_token
D:\python_project\FYP\.venv\Scripts\python.exe scripts/train_vqvae.py --config configs/exp/m0_vqvae.yaml
```

Useful overrides:

- `--device cpu` to force CPU training.
- `--epochs 10` for quick smoke tests.
- `--save-dir ./outputs/debug_run` to redirect checkpoints.

Training saves to `outputs/m0_vqvae/` by default:

- `checkpoints/encoder.pt`, `quantizer.pt`, `decoder.pt`
- `checkpoints/epoch_XXX.pt` full checkpoints
- `recon_samples/epoch_XXX.png` reconstruction previews
- `tokens/` (reserved for exported tokens)

TensorBoard logs can be added by pointing TensorBoard at the `save_dir`.

## Evaluation & Token Export

Evaluate a checkpoint on the validation split:

```bash
python scripts/evaluate_vqvae.py --config configs/exp/m0_vqvae.yaml --checkpoint outputs/m0_vqvae/checkpoints/epoch_030.pt
```

Export discrete token sequences:

```bash
python scripts/export_tokens.py \
  --config configs/exp/m0_vqvae.yaml \
  --checkpoint outputs/m0_vqvae/checkpoints/epoch_030.pt \
  --split test_manifest \
  --save-recon
```

Tokens are stored as `.npy` arrays in `outputs/m0_vqvae/tokens/`.

## Project Structure

```
lip_token/
├─ configs/
│  ├─ data/default.yaml          # Data loading knobs
│  ├─ model/resnet3d_vqvae.yaml  # Model hyperparameters
│  ├─ train/default.yaml         # Training hyperparameters
│  └─ exp/m0_vqvae.yaml          # Experiment composition
├─ src/
│  ├─ datamodules/               # Dataset, transforms, video readers
│  ├─ engine/                    # Trainer, evaluator, utilities
│  ├─ models/                    # Encoder, VQ, decoder, and future heads
│  ├─ utils/                     # Seeding, metrics, visualization
│  └─ cli/                       # CLI entrypoints
├─ scripts/                      # Convenience launch scripts
├─ outputs/                      # Checkpoints, recon samples, tokens
└─ README.md
```

## Future Extensions

- **M1**: `src/models/encoders/flow_encoder.py` — add optical-flow encoder branch.
- **M2**: `src/losses/flow_consistency.py` — enforce RGB/flow consistency.
- **M4**: `src/models/heads/ctc_head.py` — attach CTC or Transformer decoder.
- **M5**: `src/models/fusion/av_fusion.py` — fuse audio/text modalities.

All placeholder modules are present to ease incremental development.

## Notes

- Mixed precision (AMP) and multi-GPU scaling are wired for easy extension.
- Dataloaders support Decord, PyAV, or Torchvision backends.
- Reconstruction figures compare original vs. reconstructed frames (first, middle, last) for quick qualitative checks.

