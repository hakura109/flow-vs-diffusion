# LOGBOOK

> Record daily progress, decisions, and pitfalls. Newest entries on top.

## 2026-06-08 — Cloud GPU baselines: AE + DDPM

Ran both baselines to convergence on cloud GPU. Both pipelines work end to end.

- **Autoencoder (50 epochs)** — test set: PSNR **40.01** / SSIM **0.9918** / LPIPS **0.0003**.
  Near-perfect reconstruction. Caveat: the latent is *overcomplete* (8x8x128 = 8192 dims vs
  3x32x32 = 3072 input dims), i.e. it is not actually compressing. So this is a
  **no-compression upper bound**, not a meaningful bottleneck baseline. The real study varies
  the bottleneck (patch size, latent dim); this number only says "with no compression, decoding
  is essentially lossless."
- **DDPM (30 epochs, T=1000)** — train loss converged to ~**0.032**. Unconditional samples are
  rough but structured CIFAR-style images (recognizable shapes/colors, not noise). Denoising
  reconstruction at t_start=500: PSNR **13.4** / SSIM **0.20** / LPIPS **0.14**. These are low
  *by design* — at half-noise the task is largely generative (most detail destroyed, then
  re-imagined), so they are NOT directly comparable to the AE numbers above.

Conclusion: both baselines run and behave as expected. They are reference points, not a fair
head-to-head — the real diffusion vs flow-matching comparison happens later under one shared
pipeline (same backbone, same data, same eval protocol).

### Next
- [ ] Phase B: transformer encoder (per-patch hidden states) + generative head conditioned on
  each patch's hidden state — pending the "Frontier-Core" backbone reference + advisor sign-off.
- [ ] Flow-matching head as the counterpart to the DDPM head, then the fair comparison.

## 2026-06-07 — Phase A: minimal whole-image DDPM baseline (code + CPU checks)

Built the Stage 2 stepping-stone: a standard *unconditional* DDPM on full 32x32 images
(no transformer backbone, no per-patch conditioning yet). Goal was only to get the
diffusion mechanism + train/sample/eval pipeline running end to end.

- `src/models/diffusion.py`: `linear_beta_schedule(T)`; a time-conditioned `SmallUNet`
  (sinusoidal timestep embedding + skip-connected 32->16->8->16->32, ~3.34M params);
  and a `Diffusion` module with `q_sample` (closed-form forward noising), `loss`
  (epsilon-prediction MSE at a random t), `sample` (reverse chain from pure noise), and
  `reconstruct` (noise to t_start then denoise back — gives a *paired* target so the
  recon metrics are meaningful for an otherwise unconditional model).
- `scripts/train_diffusion.py`: mirrors `train_ae.py` — `--smoke` / `--overfit` / full
  training; loss to TensorBoard, an unconditional `sample()` grid, and denoising-
  reconstruction PSNR/SSIM/LPIPS on the test set. Timestamped `experiments/<ts>_diffusion_<mode>/`.

CPU checks (both pass):
- **smoke** (T=50, batch=2): shapes match, loss finite, reverse chain runs.
- **overfit** (T=1000, 32 real images, 300 steps): windowed-average loss
  `0.3308 -> 0.0486` (~6.8x drop). Per-step loss is noisy because t is resampled every
  step, so the overfit assertion compares first-window vs last-window averages rather than
  single steps.

Decisions:
- **Eval = denoising reconstruction**: an unconditional DDPM has no paired target for a
  from-noise sample, so test metrics noise each real image to `t_start = recon_t_frac * T`
  (default half) and denoise back, then compare to the original. The from-noise grid is for
  eyeballing sample quality only.
- Defaults follow DDPM convention (`lr=2e-4`, `epochs=30`, `eval-images=64` since sampling
  is slow). Full training is a cloud-GPU job; locally we only run `--smoke` / `--overfit`.

### Next
- [ ] Cloud GPU: full DDPM training, then log sample-grid quality + denoising-recon
  PSNR/SSIM/LPIPS as a separate baseline entry.

## 2026-06-07 — Stage 1 baseline on real CIFAR-10 (CPU run)

Ran the full Stage 1 pipeline locally (CPU, `flowproj` env) to lock in the AE baseline.

- **Overfit check**: `python scripts/train_ae.py --overfit` — loss `0.374692 -> 0.001560`
  over 300 steps on one 32-image batch. Passes the <10%-of-initial assertion; model learns.
- **Full training**: `python scripts/train_ae.py --epochs 5` — per-epoch train loss
  `0.0203 -> 0.0034`. Test-set metrics over 512 images:

  | Metric | Value     |
  | ------ | --------- |
  | PSNR   | 32.245 dB |
  | SSIM   | 0.9662    |
  | LPIPS  | 0.0026    |

  Outputs saved under `experiments/20260607_215354_train/`: `autoencoder.pt`,
  `recon_grid.png` (top originals / bottom reconstructions — visually near-identical),
  `metrics.md`, and TensorBoard logs. Note `experiments/` is git-ignored, so only the
  numbers live here in the LOGBOOK.

This is the **AE upper-bound reference** for the reconstruction task: it bounds how well a
clean 4x-downsampling latent can be decoded, before any generative head is involved.

Pitfalls:
- On Windows use the `flowproj` env interpreter directly
  (`C:\Users\strag\.conda\envs\flowproj\python.exe`); the base env is separate.
- Harmless warnings on this run: a NumPy 2.4 `VisibleDeprecationWarning` from torchvision's
  CIFAR pickle loader, and torchvision `pretrained`/`weights` deprecation warnings from LPIPS
  loading AlexNet. Neither affects results.

### Next
- [ ] **Phase A stepping-stone**: simplified DDPM — whole-image, pure denoising (no transformer
  backbone yet). Goal is to get the diffusion mechanism + train/sample/eval pipeline running;
  smoke test first.
- [ ] Phase B: transformer encoder (per-patch hidden states) + generative head conditioned on
  each patch's hidden state. Pending the "Frontier-Core" backbone reference + advisor sign-off.
- [ ] Cloud GPU full AE training to refine the baseline (longer schedule than 5 CPU epochs).

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
