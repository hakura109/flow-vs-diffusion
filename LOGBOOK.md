# LOGBOOK

> Record daily progress, decisions, and pitfalls. Newest entries on top.

## 2026-06-01 — FFHQ-64 data pipeline

- `src/data/datasets.py` adds `FlatImageDataset` + `get_ffhq64_dataset/dataloader`:
  recursively reads images from a local folder (ImageFolder-style, no class subdirs
  needed), resizes the short side then center-crops to 64x64, and normalizes to [-1,1].
  The interface matches `get_cifar10_dataloader` (FFHQ has no built-in train/test split,
  so `train` only controls shuffle/drop_last).
- `scripts/make_ffhq64_placeholders.py`: generates a few dozen 64x64 placeholder images
  before real data is available, so the pipeline runs; drop real FFHQ-64 into
  `data/ffhq64/` to replace them, no loader changes needed.
- Verified: 48 placeholders -> batch `(16,3,64,64)`, range [-1.000, 0.992].
- Note: `data/` is git-ignored, so images are not committed.

## 2026-06-01 — Stage 1: AE running on real CIFAR-10

Completed `scripts/train_ae.py` with three additions:

- **Full training**: real CIFAR-10 (auto-downloads to `data/` on first run, ~170MB),
  MSE + Adam, per-step loss written to TensorBoard (`experiments/<timestamp>_train/tb`),
  weights saved as `autoencoder.pt` after training.
- **`--overfit` mode**: take one real batch (32 images by default) and train it for 300
  steps to verify the model actually learns.
  - Result: loss `0.3747 -> 0.00156` (about a 240x drop), ~1 minute on CPU.
  - Assertions: final loss < 10% of the initial value, and < 0.01 counts as close to 0.
- **Evaluation + visualization**: after training, compute mean PSNR/SSIM/LPIPS on the
  test set, save a reconstruction grid `recon_grid.png` (top row originals, bottom row
  reconstructions) and a `metrics.md` table, all under `experiments/<timestamp>_train/`.

Conventions and pitfalls:
- Three modes selected by flag: `--smoke` (synthetic images) / `--overfit` (one real batch) / default full training.
- LPIPS expects inputs in `[-1, 1]`, matching the data normalization; PSNR/SSIM convert back to `[0, 1]` internally.
- Full training is done on cloud GPU only; locally we just use `--overfit` and short epochs to verify the pipeline doesn't crash.

### TODO
- [ ] Full AE training on cloud GPU and record baseline PSNR/SSIM/LPIPS
- [ ] Diffusion reconstruction baseline
- [ ] Flow matching reconstruction baseline
- [ ] Unified evaluation script and comparison table

## 2026-06-01 — Project scaffold

- Created conda env `flowproj` (python 3.11) and installed CPU-build dependencies.
- Set up the directory structure and basic utilities:
  - `src/utils/setup.py`: `set_seed` / `get_device`
  - `src/data/datasets.py`: CIFAR-10 dataloader, normalized to [-1, 1]
  - `src/data/patchify.py`: einops-based patchify / unpatchify + roundtrip test
  - `src/metrics/image_metrics.py`: psnr / ssim / lpips
  - `src/models/autoencoder.py`: small conv autoencoder (Tanh output, [-1, 1])
  - `scripts/train_ae.py`: training entry point, supports `--smoke`
- Made the first git commit.
- Smoke test `python scripts/train_ae.py --smoke` passes.

### TODO
- [ ] Diffusion reconstruction baseline
- [ ] Flow matching reconstruction baseline
- [ ] Unified evaluation script and comparison table
