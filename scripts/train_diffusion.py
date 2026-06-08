"""DDPM training entry point (whole-image, unconditional). Mirrors scripts/train_ae.py.

Three modes:
    python scripts/train_diffusion.py --smoke    # CPU smoke test: batch=2, tiny T, one fwd+bwd + a quick sample
    python scripts/train_diffusion.py --overfit  # repeat one real batch to verify the loss trends down
    python scripts/train_diffusion.py            # full training on real CIFAR-10 (auto-downloads on first run)

Full training writes the loss to TensorBoard, then after training saves an unconditional sample
grid via Diffusion.sample(), and reports denoising-reconstruction PSNR/SSIM/LPIPS on the test set
(see evaluate() for why an unconditional DDPM is scored this way). Everything for a run lands under
experiments/<timestamp>_<mode>/, so old results are never overwritten.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# Let the script import the project's src package directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import denormalize, get_cifar10_dataloader, get_ffhq64_dataloader
from src.metrics.image_metrics import lpips, psnr, ssim
from src.models.diffusion import Diffusion, SmallUNet
from src.utils.setup import get_device, set_seed

EXPERIMENTS = ROOT / "experiments"
IMAGE_SIZE = {"cifar10": 32, "ffhq64": 64}  # spatial size per dataset (the U-Net adapts automatically)


def make_run_dir(dataset: str, mode: str, timestamp: str) -> Path:
    run_dir = EXPERIMENTS / f"{timestamp}_diffusion_{dataset}_{mode}"
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_loader(args: argparse.Namespace, train: bool, batch_size: int, shuffle=None):
    """Return a dataloader for the chosen dataset; images are normalized to [-1, 1] either way."""
    if args.dataset == "ffhq64":
        # FFHQ has no train/test split, so `train` only toggles shuffle/drop_last (see datasets.py).
        return get_ffhq64_dataloader(
            root=args.ffhq_root,
            train=train,
            batch_size=batch_size,
            shuffle=shuffle,
            image_size=64,
        )
    return get_cifar10_dataloader(
        root=args.data_root, train=train, batch_size=batch_size, shuffle=shuffle
    )


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def smoke_test() -> None:
    print("=== DDPM smoke test ===")
    set_seed(42)
    device = torch.device("cpu")  # smoke test forces CPU
    small_T = 50  # temporarily tiny schedule so the reverse loop is cheap
    print(f"device: {device}  timesteps(T)={small_T}")

    # 2 synthetic 32x32 RGB images in [-1, 1] (no data download required).
    x = torch.rand(2, 3, 32, 32, device=device) * 2 - 1
    print(f"input shape : {tuple(x.shape)}  range=[{x.min():.3f}, {x.max():.3f}]")

    model = SmallUNet().to(device)
    diffusion = Diffusion(timesteps=small_T).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    # Shape check: the noise prediction must match the input shape.
    t = torch.randint(0, small_T, (x.shape[0],), device=device)
    x_t = diffusion.q_sample(x, t)
    pred = model(x_t, t)
    print(f"x_t shape   : {tuple(x_t.shape)}   pred shape: {tuple(pred.shape)}")
    assert pred.shape == x.shape, "noise prediction shape does not match input"

    # One forward + backward through the training loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = diffusion.loss(model, x)
    loss.backward()
    optimizer.step()
    print(f"loss        : {loss.item():.6f}")
    assert torch.isfinite(loss), "loss is not finite"

    # Quick sample so the reverse chain is exercised too (2 images, tiny T -> fast on CPU).
    samples = diffusion.sample(model, (2, 3, 32, 32))
    print(f"sample shape: {tuple(samples.shape)}  range=[{samples.min():.3f}, {samples.max():.3f}]")
    assert samples.shape == x.shape, "sample shape does not match input"

    print("smoke test PASSED")


# --------------------------------------------------------------------------- #
# overfit: repeat a single batch to verify the loss trends down
# --------------------------------------------------------------------------- #
def overfit(args: argparse.Namespace) -> None:
    print("=== DDPM overfit (repeat a single batch) ===")
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}  timesteps(T)={args.timesteps}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = make_run_dir(args.dataset, "overfit", timestamp)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    loader = build_loader(args, train=True, batch_size=args.overfit_batch, shuffle=True)
    images, _ = next(iter(loader))
    images = images.to(device)
    print(f"overfit on {images.shape[0]} real images for {args.overfit_steps} steps")

    model = SmallUNet().to(device)
    diffusion = Diffusion(timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    losses: list[float] = []
    for step in range(args.overfit_steps):
        optimizer.zero_grad()
        loss = diffusion.loss(model, images)  # t is resampled each step, so the loss is noisy
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        writer.add_scalar("overfit/loss", loss.item(), step)
        if step % 50 == 0 or step == args.overfit_steps - 1:
            print(f"  step {step:4d}/{args.overfit_steps}  loss={loss.item():.6f}")

    writer.close()

    # The per-step loss is noisy (random t each step), so compare windowed averages, not single steps.
    window = max(1, args.overfit_steps // 10)
    first_avg = sum(losses[:window]) / window
    last_avg = sum(losses[-window:]) / window
    print(f"avg of first {window} = {first_avg:.6f}  ->  avg of last {window} = {last_avg:.6f}")

    assert torch.isfinite(torch.tensor(last_avg)), "loss is not finite"
    assert last_avg < first_avg, "loss did not trend down; model may not be learning"
    print(f"overfit PASSED  loss trended down ({first_avg:.6f} -> {last_avg:.6f})")
    print(f"TensorBoard logs: {run_dir / 'tb'}")


# --------------------------------------------------------------------------- #
# evaluation + visualization
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(diffusion: Diffusion, model, loader, device, max_images: int, t_start: int):
    """Denoising-reconstruction metrics over the first max_images of the test set.

    An unconditional DDPM has no paired target for a from-noise sample, so we instead noise each
    real image up to t_start and denoise it back, then compare to the original. Larger t_start =
    harder reconstruction (more information destroyed before denoising).
    """
    model.eval()
    tot_psnr = tot_ssim = tot_lpips = 0.0
    n = 0
    for images, _ in loader:
        images = images.to(device)
        recon = diffusion.reconstruct(model, images, t_start, device=device)
        bs = images.shape[0]
        tot_psnr += psnr(recon, images).item() * bs
        tot_ssim += ssim(recon, images).item() * bs
        tot_lpips += lpips(recon, images).item() * bs
        n += bs
        if n >= max_images:
            break
    return {
        "PSNR": tot_psnr / n,
        "SSIM": tot_ssim / n,
        "LPIPS": tot_lpips / n,
        "n_images": n,
        "t_start": t_start,
    }


@torch.no_grad()
def save_sample_grid(diffusion: Diffusion, model, n: int, path: Path, device, image_size: int) -> None:
    """Generate n images from pure noise and save them as a single grid (for a quick eyeball)."""
    model.eval()
    samples = diffusion.sample(model, (n, 3, image_size, image_size), device=device)
    grid = make_grid(denormalize(samples), nrow=n, padding=2)
    save_image(grid, str(path))


def format_metrics_table(metrics: dict) -> str:
    lines = [
        "| Metric | Value |",
        "| ------ | ----- |",
        f"| PSNR    | {metrics['PSNR']:.3f} dB |",
        f"| SSIM    | {metrics['SSIM']:.4f} |",
        f"| LPIPS   | {metrics['LPIPS']:.4f} |",
        f"| t_start | {metrics['t_start']} |",
        f"| images  | {metrics['n_images']} |",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# full training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> None:
    print(f"=== DDPM full training ({args.dataset}) ===")
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}  timesteps(T)={args.timesteps}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = make_run_dir(args.dataset, "train", timestamp)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    train_loader = build_loader(args, train=True, batch_size=args.batch_size)
    test_loader = build_loader(args, train=False, batch_size=args.batch_size)

    model = SmallUNet().to(device)
    diffusion = Diffusion(timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = diffusion.loss(model, images)
            loss.backward()
            optimizer.step()

            running += loss.item()
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        epoch_loss = running / len(train_loader)
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={epoch_loss:.6f}")

    # Save weights.
    ckpt_path = run_dir / "diffusion_unet.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Unconditional sample grid (from pure noise).
    grid_path = run_dir / "sample_grid.png"
    save_sample_grid(diffusion, model, args.grid_images, grid_path, device, IMAGE_SIZE[args.dataset])

    # Denoising-reconstruction metrics on the test set.
    t_start = int(args.recon_t_frac * args.timesteps)
    metrics = evaluate(diffusion, model, test_loader, device, args.eval_images, t_start)
    for k in ("PSNR", "SSIM", "LPIPS"):
        writer.add_scalar(f"test/{k}", metrics[k], 0)

    table = format_metrics_table(metrics)
    (run_dir / "metrics.md").write_text(
        f"# DDPM evaluation\n\nUnconditional samples: `sample_grid.png`\n\n"
        f"Denoising reconstruction (noise to t_start, then denoise back):\n\n{table}\n",
        encoding="utf-8",
    )
    writer.close()

    print("\nTest-set denoising-reconstruction:")
    print(table)
    print(f"\nSample grid:   {grid_path}")
    print(f"Weights:       {ckpt_path}")
    print(f"Metrics table: {run_dir / 'metrics.md'}")
    print(f"TensorBoard:   {run_dir / 'tb'}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an unconditional DDPM on CIFAR-10")
    p.add_argument("--smoke", action="store_true", help="CPU smoke test: 2 images, tiny T, one fwd+bwd + quick sample")
    p.add_argument("--overfit", action="store_true", help="repeat a single batch to verify the loss trends down")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=["cifar10", "ffhq64"], default="cifar10", help="dataset to use")
    p.add_argument("--data-root", type=str, default="data", help="root for CIFAR-10 (--dataset cifar10)")
    p.add_argument("--ffhq-root", type=str, default="data/ffhq64", help="folder of FFHQ-64 PNGs (--dataset ffhq64)")
    p.add_argument("--timesteps", type=int, default=1000, help="number of diffusion steps T")
    # overfit-related
    p.add_argument("--overfit-steps", type=int, default=300)
    p.add_argument("--overfit-batch", type=int, default=32)
    # evaluation/visualization-related
    p.add_argument("--eval-images", type=int, default=64, help="test images averaged into the metrics (sampling is slow)")
    p.add_argument("--grid-images", type=int, default=8, help="number of images in the sample grid")
    p.add_argument("--recon-t-frac", type=float, default=0.5, help="reconstruction starts at t = frac * T")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke_test()
    elif args.overfit:
        overfit(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
