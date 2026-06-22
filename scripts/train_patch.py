"""Patch-transformer training entry point. Mirrors scripts/train_diffusion.py.

Trains PatchReconstructionModel: a transformer backbone encodes an image into per-patch hidden
states (the bottleneck), and a per-patch conditional generative head reconstructs each clean patch
from noise. The head is selectable with --head:
    --head diffusion   (DiffusionPatchHead, DDPM-style)
    --head flow        (FlowMatchingPatchHead, rectified-flow / Euler)
Both share the same backbone, dataloaders, metrics, and eval protocol, so the two are a fair,
capacity-matched comparison (only the generative mechanism differs).

Three modes:
    python scripts/train_patch.py --smoke              # CPU: batch=2, tiny head, one fwd+bwd + a quick reconstruct
    python scripts/train_patch.py --overfit            # repeat one real batch, confirm the loss trends down
    python scripts/train_patch.py                      # full training on real data (auto-downloads CIFAR-10)
    python scripts/train_patch.py --head flow --dataset ffhq64

Full training writes the loss to TensorBoard, then saves a reconstruction grid (encode -> sample
patches from noise given cond -> unpatchify) and reports reconstruction PSNR/SSIM/LPIPS on the test
set. Everything for a run lands under experiments/<timestamp>_patch_<head>_<dataset>_<mode>/.
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
from src.models.patch_transformer import PatchReconstructionModel
from src.utils.setup import get_device, set_seed

EXPERIMENTS = ROOT / "experiments"
IMAGE_SIZE = {"cifar10": 32, "ffhq64": 64}  # spatial size per dataset (the backbone adapts via img_size)


def make_run_dir(dataset: str, head: str, mode: str, timestamp: str) -> Path:
    run_dir = EXPERIMENTS / f"{timestamp}_patch_{head}_{dataset}_{mode}"
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_loader(args: argparse.Namespace, train: bool, batch_size: int, shuffle=None):
    """Return a dataloader for the chosen dataset; images are normalized to [-1, 1] either way."""
    if args.dataset == "ffhq64":
        # FFHQ has no train/test split, so `train` only toggles shuffle/drop_last (see datasets.py).
        return get_ffhq64_dataloader(
            root=args.ffhq_root, train=train, batch_size=batch_size, shuffle=shuffle, image_size=64
        )
    return get_cifar10_dataloader(
        root=args.data_root, train=train, batch_size=batch_size, shuffle=shuffle
    )


def build_model(args: argparse.Namespace, image_size: int, device: torch.device) -> PatchReconstructionModel:
    """Construct PatchReconstructionModel with the chosen head; backbone/head capacity are shared."""
    return PatchReconstructionModel(
        img_size=image_size,
        patch_size=args.patch_size,
        channels=3,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        head_kind=args.head,
        timesteps=args.timesteps,        # used by the diffusion head
        head_hidden=args.head_hidden,
        head_blocks=args.head_blocks,
        flow_sample_steps=args.flow_steps,  # used by the flow head
    ).to(device)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def smoke_test(args: argparse.Namespace) -> None:
    print(f"=== patch model smoke test (head={args.head}) ===")
    set_seed(42)
    device = torch.device("cpu")  # smoke test forces CPU
    print(f"device: {device}")

    # 2 synthetic 32x32 RGB images in [-1, 1] (no data download required).
    x = torch.rand(2, 3, 32, 32, device=device) * 2 - 1
    print(f"input shape : {tuple(x.shape)}  range=[{x.min():.3f}, {x.max():.3f}]")

    # Tiny generative schedule so the reverse chain / Euler integration is cheap on CPU.
    model = PatchReconstructionModel(
        img_size=32, patch_size=args.patch_size, channels=3,
        dim=args.dim, depth=args.depth, num_heads=args.num_heads,
        head_kind=args.head, head_hidden=args.head_hidden, head_blocks=args.head_blocks,
        timesteps=20, flow_sample_steps=8,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}  (backbone + {args.head} head)")

    # One forward + backward through the training loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = model.loss(x)
    loss.backward()
    optimizer.step()
    print(f"loss        : {loss.item():.6f}")
    assert loss.dim() == 0 and torch.isfinite(loss), "loss is not a finite scalar"

    # Quick reconstruct so the head's sampling path is exercised too.
    recon = model.reconstruct(x)
    print(f"recon shape : {tuple(recon.shape)}  range=[{recon.min():.3f}, {recon.max():.3f}]")
    assert recon.shape == x.shape, "reconstruction shape does not match input"
    assert torch.isfinite(recon).all(), "reconstruction is not finite"
    assert recon.min() >= -1.0 and recon.max() <= 1.0, "reconstruction not in [-1, 1]"

    print("smoke test PASSED")


# --------------------------------------------------------------------------- #
# overfit: repeat a single batch to verify the loss trends down
# --------------------------------------------------------------------------- #
def overfit(args: argparse.Namespace) -> None:
    print(f"=== patch model overfit (head={args.head}, repeat a single batch) ===")
    set_seed(args.seed)
    device = get_device()
    image_size = IMAGE_SIZE[args.dataset]
    print(f"device: {device}  head: {args.head}  dataset: {args.dataset}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = make_run_dir(args.dataset, args.head, "overfit", timestamp)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    loader = build_loader(args, train=True, batch_size=args.overfit_batch, shuffle=True)
    images, _ = next(iter(loader))
    images = images.to(device)
    print(f"overfit on {images.shape[0]} real images for {args.overfit_steps} steps")

    model = build_model(args, image_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    losses: list[float] = []
    for step in range(args.overfit_steps):
        optimizer.zero_grad()
        loss = model.loss(images)  # both heads resample t (and noise) each step, so the loss is noisy
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        writer.add_scalar("overfit/loss", loss.item(), step)
        if step % 50 == 0 or step == args.overfit_steps - 1:
            print(f"  step {step:4d}/{args.overfit_steps}  loss={loss.item():.6f}")

    writer.close()

    # The per-step loss is noisy (random t/noise each step), so compare windowed averages.
    window = max(1, args.overfit_steps // 10)
    first_avg = sum(losses[:window]) / window
    last_avg = sum(losses[-window:]) / window
    print(f"avg of first {window} = {first_avg:.6f}  ->  avg of last {window} = {last_avg:.6f}")

    assert torch.isfinite(torch.tensor(last_avg)), "loss is not finite"
    assert last_avg < first_avg, "loss did not trend down; model may not be learning"
    if last_avg < 0.05:
        print(f"overfit PASSED  loss dropped close to 0 ({first_avg:.6f} -> {last_avg:.6f})")
    else:
        print(f"overfit OK  loss trended down a lot but not below 0.05 ({first_avg:.6f} -> {last_avg:.6f}); "
              f"consider more steps")
    print(f"TensorBoard logs: {run_dir / 'tb'}")


# --------------------------------------------------------------------------- #
# evaluation + visualization
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: PatchReconstructionModel, loader, device, max_images: int):
    """Reconstruction PSNR/SSIM/LPIPS over the first max_images of the test set.

    Reconstruction = encode each image to per-patch hidden states, then sample each patch from pure
    noise conditioned on its hidden state, and compare the unpatchified image to the original.
    """
    model.eval()
    tot_psnr = tot_ssim = tot_lpips = 0.0
    n = 0
    for images, _ in loader:
        images = images.to(device)
        recon = model.reconstruct(images)
        bs = images.shape[0]
        tot_psnr += psnr(recon, images).item() * bs
        tot_ssim += ssim(recon, images).item() * bs
        tot_lpips += lpips(recon, images).item() * bs
        n += bs
        if n >= max_images:
            break
    return {"PSNR": tot_psnr / n, "SSIM": tot_ssim / n, "LPIPS": tot_lpips / n, "n_images": n}


@torch.no_grad()
def save_recon_grid(model: PatchReconstructionModel, images: torch.Tensor, path: Path, n: int = 8) -> None:
    """Save a grid: top row originals, bottom row reconstructions (sampled from noise given cond)."""
    model.eval()
    orig = images[:n]
    recon = model.reconstruct(orig)
    both = torch.cat([denormalize(orig), denormalize(recon)], dim=0)  # 2n
    grid = make_grid(both, nrow=n, padding=2)
    save_image(grid, str(path))


def format_metrics_table(metrics: dict) -> str:
    lines = [
        "| Metric | Value |",
        "| ------ | ----- |",
        f"| PSNR   | {metrics['PSNR']:.3f} dB |",
        f"| SSIM   | {metrics['SSIM']:.4f} |",
        f"| LPIPS  | {metrics['LPIPS']:.4f} |",
        f"| images | {metrics['n_images']} |",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# full training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> None:
    print(f"=== patch training ({args.dataset}, head={args.head}) ===")
    set_seed(args.seed)
    device = get_device()
    image_size = IMAGE_SIZE[args.dataset]
    print(f"device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = make_run_dir(args.dataset, args.head, "train", timestamp)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    train_loader = build_loader(args, train=True, batch_size=args.batch_size)
    test_loader = build_loader(args, train=False, batch_size=args.batch_size)

    model = build_model(args, image_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}  (backbone + {args.head} head)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = model.loss(images)
            loss.backward()
            optimizer.step()

            running += loss.item()
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        epoch_loss = running / len(train_loader)
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={epoch_loss:.6f}")

    # Save weights.
    ckpt_path = run_dir / f"patch_{args.head}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Reconstruction grid (sampled from noise given cond).
    vis_images, _ = next(iter(test_loader))
    grid_path = run_dir / "recon_grid.png"
    save_recon_grid(model, vis_images.to(device), grid_path, n=args.grid_images)

    # Reconstruction metrics on the test set.
    metrics = evaluate(model, test_loader, device, max_images=args.eval_images)
    for k in ("PSNR", "SSIM", "LPIPS"):
        writer.add_scalar(f"test/{k}", metrics[k], 0)

    table = format_metrics_table(metrics)
    (run_dir / "metrics.md").write_text(
        f"# Patch reconstruction evaluation (head={args.head}, {args.dataset})\n\n"
        f"Top row originals, bottom row reconstructions: `recon_grid.png`\n\n"
        f"Reconstruction = encode -> sample each patch from noise given its hidden state.\n\n{table}\n",
        encoding="utf-8",
    )
    writer.close()

    print("\nTest-set reconstruction:")
    print(table)
    print(f"\nReconstruction grid: {grid_path}")
    print(f"Weights:             {ckpt_path}")
    print(f"Metrics table:       {run_dir / 'metrics.md'}")
    print(f"TensorBoard:         {run_dir / 'tb'}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PatchReconstructionModel (diffusion or flow head)")
    p.add_argument("--smoke", action="store_true", help="CPU smoke test: 2 images, tiny head, one fwd+bwd + reconstruct")
    p.add_argument("--overfit", action="store_true", help="repeat a single batch to verify the loss trends down")
    p.add_argument("--head", choices=["diffusion", "flow"], default="diffusion", help="generative head")
    p.add_argument("--dataset", choices=["cifar10", "ffhq64"], default="cifar10", help="dataset to use")
    p.add_argument("--data-root", type=str, default="data", help="root for CIFAR-10 (--dataset cifar10)")
    p.add_argument("--ffhq-root", type=str, default="data/ffhq64", help="folder of FFHQ-64 PNGs (--dataset ffhq64)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    # backbone / head architecture (shared across heads for a fair comparison)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--dim", type=int, default=128, help="transformer token dimension D")
    p.add_argument("--depth", type=int, default=4, help="number of transformer blocks")
    p.add_argument("--num-heads", type=int, default=4, help="attention heads")
    p.add_argument("--head-hidden", type=int, default=256, help="generative head MLP width")
    p.add_argument("--head-blocks", type=int, default=3, help="generative head residual blocks")
    # head-specific generative knobs
    p.add_argument("--timesteps", type=int, default=1000, help="diffusion head: number of diffusion steps T")
    p.add_argument("--flow-steps", type=int, default=50, help="flow head: number of Euler integration steps")
    # overfit-related
    p.add_argument("--overfit-steps", type=int, default=500)
    p.add_argument("--overfit-batch", type=int, default=8)
    # evaluation/visualization-related
    p.add_argument("--eval-images", type=int, default=64, help="test images averaged into the metrics (sampling is slow)")
    p.add_argument("--grid-images", type=int, default=8, help="images per row in the reconstruction grid")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke_test(args)
    elif args.overfit:
        overfit(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
