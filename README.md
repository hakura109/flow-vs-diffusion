# Flow Matching vs Diffusion: Image Patch Reconstruction

Graduation project: comparing the **Diffusion** and **Flow Matching** generative
paradigms on an image patch reconstruction task.

## Goal

Under the same data (CIFAR-10), the same backbone, and the same evaluation protocol,
make a fair comparison between diffusion and flow matching on image patch
reconstruction quality (PSNR / SSIM / LPIPS).

## Workflow

- **Local (Windows, CPU)**: smoke tests only, to confirm the code doesn't crash and that shapes and loss look right.
- **Cloud (GPU)**: the real training and evaluation.

## Environment

```bash
conda create -n flowproj python=3.11 -y
conda activate flowproj
# CPU build of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Directory structure

```
configs/        experiment configs (yaml)
src/
  data/         datasets and patchify
  models/       network architectures (autoencoder / diffusion / flow)
  metrics/      evaluation metrics (psnr/ssim/lpips)
  train/        training loops
  utils/        common utilities (seed/device)
scripts/        runnable entry-point scripts
experiments/    training outputs (git-ignored)
notebooks/      exploratory analysis
```

## Quick check

```bash
python scripts/train_ae.py --smoke
```

The smoke test takes 2 images, runs one forward + backward pass, and prints the output shape and loss.

See [LOGBOOK.md](LOGBOOK.md) for detailed progress.
