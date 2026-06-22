"""Sweep NFE (number of function evaluations = sampling steps) for a trained patch model.

Loads a trained `PatchReconstructionModel` checkpoint and, on a FIXED test subset, reconstructs the
images at several NFE budgets, recording reconstruction quality (PSNR/SSIM/LPIPS) and wall-clock time
at each budget. This is the speed/quality trade-off curve: how few sampling steps can the head get
away with before quality falls off.

What "NFE" means per head (one NFE = one forward pass of the per-patch generative network):
  - flow      : forward-Euler steps. Uses `head.sample(cond, num_steps=NFE)` directly -- the flow
                head already exposes a per-call step override, so NFE = Euler steps = velocity-net calls.
  - diffusion : the diffusion head's native sample() runs the FULL T-step DDPM ancestral chain with no
                knob to shorten it, so we sample it with deterministic DDIM (eta=0) respacing instead:
                pick NFE timesteps spaced over [0, T-1] and integrate. NFE = denoiser calls, matching
                the flow head's step count exactly so the two are compared on the same NFE axis.
                NOTE: DDIM is a *different* sampler from the training-time eval (full DDPM ancestral),
                so the diffusion numbers here are a within-head NFE curve, not a restatement of the
                train_patch.py eval.

Timing protocol: for each NFE we run one un-timed warmup, then time `--timing-runs` (default 3)
reconstructions and report the mean (with a CUDA sync around each timed run on GPU). Quality is
measured on one seed-fixed reconstruction so the only thing that varies across NFE is the step count.

Usage:
    # CPU smoke: random tiny model + small NFEs, no checkpoint / no data download
    python scripts/sweep_nfe.py --smoke

    # sweep a trained flow head, save a markdown table + the two curves
    python scripts/sweep_nfe.py --head flow \
        --ckpt experiments/<ts>_patch_flow_cifar10_train/patch_flow.pt \
        --dataset cifar10 --eval-images 64 --plot

    # diffusion control (sampled with DDIM respacing)
    python scripts/sweep_nfe.py --head diffusion \
        --ckpt experiments/<ts>_patch_diffusion_cifar10_train/patch_diffusion.pt --plot

Outputs land in experiments/<timestamp>_nfe_<head>_<dataset>/ (nfe_sweep.md [+ nfe_sweep.png]).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch

# Let the script import the project's src package directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import get_cifar10_dataloader, get_ffhq64_dataloader
from src.data.patchify import unpatchify
from src.metrics.image_metrics import lpips, psnr, ssim
from src.models.patch_transformer import (
    DiffusionPatchHead,
    FlowMatchingPatchHead,
    PatchReconstructionModel,
)
from src.utils.setup import get_device, set_seed

EXPERIMENTS = ROOT / "experiments"
IMAGE_SIZE = {"cifar10": 32, "ffhq64": 64}
DEFAULT_NFES = [1, 2, 5, 10, 20, 50]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_loader(args: argparse.Namespace, train: bool, batch_size: int, shuffle=None):
    """Return a dataloader for the chosen dataset; images are normalized to [-1, 1] either way."""
    if args.dataset == "ffhq64":
        return get_ffhq64_dataloader(
            root=args.ffhq_root, train=train, batch_size=batch_size, shuffle=shuffle, image_size=64
        )
    return get_cifar10_dataloader(
        root=args.data_root, train=train, batch_size=batch_size, shuffle=shuffle
    )


def load_eval_images(args: argparse.Namespace, device: torch.device, n: int) -> torch.Tensor:
    """Load the first n test images in a fixed order (shuffle=False), so every NFE sees the same batch."""
    loader = build_loader(args, train=False, batch_size=args.batch_size, shuffle=False)
    chunks, got = [], 0
    for images, _ in loader:
        chunks.append(images)
        got += images.shape[0]
        if got >= n:
            break
    return torch.cat(chunks, dim=0)[:n].to(device)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(args: argparse.Namespace, image_size: int, device: torch.device, head_kind: str
                ) -> PatchReconstructionModel:
    """Construct a PatchReconstructionModel; architecture must match the trained checkpoint."""
    return PatchReconstructionModel(
        img_size=image_size,
        patch_size=args.patch_size,
        channels=3,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        head_kind=head_kind,
        timesteps=args.timesteps,
        head_hidden=args.head_hidden,
        head_blocks=args.head_blocks,
        flow_sample_steps=args.flow_steps,
    ).to(device)


# --------------------------------------------------------------------------- #
# NFE-controllable reconstruction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _ddim_sample(head: DiffusionPatchHead, cond: torch.Tensor, nfe: int, device: torch.device
                 ) -> torch.Tensor:
    """Deterministic DDIM (eta=0) respacing to exactly `nfe` steps for the diffusion head.

    DDIM is the standard way to sample a trained epsilon-prediction diffusion model in fewer steps:
    choose `nfe` timesteps spaced over [0, T-1] (descending from the noisiest), and at each one predict
    x0 from the noise estimate, then jump straight to the next timestep. NFE equals the number of
    denoiser() calls, so it lines up with the flow head's Euler-step count. nfe=1 is a single one-shot
    x0 prediction from pure noise. Returns clean patches (B, N, patch_dim).
    """
    alpha_bars = head.diffusion.alpha_bars  # (T,)
    timesteps = head.diffusion.timesteps
    b, n, d = cond.shape
    condf = cond.reshape(b * n, d)
    m = b * n

    x = torch.randn(m, head.patch_dim, device=device)  # pure noise at the top of the chain
    # `nfe` timesteps, descending from T-1 down to 0 (nfe=1 -> just [T-1]).
    seq = torch.linspace(timesteps - 1, 0, steps=nfe).round().long().tolist()
    for j, t_cur in enumerate(seq):
        t_batch = torch.full((m,), t_cur, device=device, dtype=torch.long)
        eps = head.denoiser(x, t_batch, condf)
        ab_cur = alpha_bars[t_cur]
        # Predicted clean patch from the current noise estimate (clamped -- helps a lot at few steps).
        x0 = ((x - torch.sqrt(1.0 - ab_cur) * eps) / torch.sqrt(ab_cur)).clamp(-1.0, 1.0)
        t_prev = seq[j + 1] if j + 1 < len(seq) else -1
        ab_prev = alpha_bars[t_prev] if t_prev >= 0 else torch.ones((), device=device)
        # DDIM deterministic update (eta=0): re-noise the x0 estimate to the next timestep.
        x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps
    return x.reshape(b, n, head.patch_dim)


@torch.no_grad()
def reconstruct_nfe(model: PatchReconstructionModel, images: torch.Tensor, nfe: int) -> torch.Tensor:
    """One reconstruction at the given NFE. encode -> sample patches at `nfe` steps -> unpatchify."""
    head = model.head
    cond = model.backbone(images)  # (B, N, D) per-patch hidden states
    if isinstance(head, FlowMatchingPatchHead):
        patches = head.sample(cond, num_steps=nfe)            # exactly nfe Euler steps
    elif isinstance(head, DiffusionPatchHead):
        patches = _ddim_sample(head, cond, nfe, images.device)  # exactly nfe DDIM steps
    else:
        raise TypeError(f"unsupported head type {type(head).__name__}")
    recon = unpatchify(patches, model.patch_size, model.img_size, model.img_size, model.channels)
    return recon.clamp(-1.0, 1.0)


# --------------------------------------------------------------------------- #
# metrics + timing
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_metrics(recon: torch.Tensor, target: torch.Tensor, chunk: int = 32) -> dict:
    """Mean PSNR/SSIM/LPIPS over the batch, chunked to bound memory. Both inputs in [-1, 1]."""
    recon = recon.clamp(-1.0, 1.0)
    n = recon.shape[0]
    tot_p = tot_s = tot_l = 0.0
    for i in range(0, n, chunk):
        r, t = recon[i:i + chunk], target[i:i + chunk]
        bs = r.shape[0]
        tot_p += psnr(r, t).item() * bs
        tot_s += ssim(r, t).item() * bs
        tot_l += lpips(r, t).item() * bs
    return {"PSNR": tot_p / n, "SSIM": tot_s / n, "LPIPS": tot_l / n}


def time_reconstruct(model: PatchReconstructionModel, images: torch.Tensor, nfe: int,
                     runs: int, device: torch.device) -> float:
    """Mean wall-clock seconds of one reconstruction: one un-timed warmup, then `runs` timed runs."""
    reconstruct_nfe(model, images, nfe)  # warmup (caches, autotune); not timed
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        reconstruct_nfe(model, images, nfe)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return mean(times)


def run_sweep(model: PatchReconstructionModel, images: torch.Tensor, nfes: list[int],
              runs: int, device: torch.device, seed: int) -> list[dict]:
    """For each NFE: seed-fixed quality metrics + warmup-then-mean timing. Returns one row per NFE."""
    model.eval()
    rows = []
    for nfe in nfes:
        set_seed(seed)  # identical initial noise across NFEs, so only the step count varies
        recon = reconstruct_nfe(model, images, nfe)
        metrics = compute_metrics(recon, images)
        secs = time_reconstruct(model, images, nfe, runs, device)
        n_img = images.shape[0]
        metrics.update({"NFE": nfe, "time_s": secs, "ms_per_img": 1000.0 * secs / n_img})
        rows.append(metrics)
        print(f"  NFE={nfe:3d}  PSNR={metrics['PSNR']:7.3f}  SSIM={metrics['SSIM']:.4f}  "
              f"LPIPS={metrics['LPIPS']:.4f}  time={secs * 1000:8.1f} ms  "
              f"({metrics['ms_per_img']:.2f} ms/img)")
    return rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def format_table(rows: list[dict]) -> str:
    lines = [
        "| NFE | PSNR (dB) | SSIM | LPIPS | time (s) | ms/img |",
        "| --- | --------- | ---- | ----- | -------- | ------ |",
    ]
    for r in rows:
        lines.append(
            f"| {r['NFE']} | {r['PSNR']:.3f} | {r['SSIM']:.4f} | {r['LPIPS']:.4f} "
            f"| {r['time_s']:.4f} | {r['ms_per_img']:.2f} |"
        )
    return "\n".join(lines)


def build_markdown(rows: list[dict], head: str, dataset: str, ckpt: str, n_images: int,
                   runs: int, device: torch.device) -> str:
    sampler = ("Flow head sampled with forward-Euler; NFE = Euler steps = velocity-net calls."
               if head == "flow" else
               "Diffusion head sampled with deterministic DDIM (eta=0) respacing; NFE = denoiser "
               "calls. This differs from the training-time full DDPM ancestral eval.")
    return (
        f"# NFE sweep ({head} head, {dataset})\n\n"
        f"Checkpoint: `{ckpt}`  \n"
        f"Fixed eval subset: **{n_images}** images | timing: 1 warmup + mean of **{runs}** runs "
        f"| device: **{device}**\n\n"
        f"Reconstruction = encode -> sample each patch from pure noise at the given NFE. {sampler}\n\n"
        f"{format_table(rows)}\n"
    )


def maybe_plot(rows: list[dict], path: Path, head: str):
    """Save two curves side by side: quality (PSNR) vs NFE and wall-clock vs NFE. Needs matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless; no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots (--plot was requested)")
        return None

    nfes = [r["NFE"] for r in rows]
    psnrs = [r["PSNR"] for r in rows]
    times = [r["time_s"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(nfes, psnrs, "o-")
    ax1.set_xscale("log")
    ax1.set_xticks(nfes)
    ax1.set_xticklabels([str(x) for x in nfes])
    ax1.set_xlabel("NFE (sampling steps)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"Quality vs NFE ({head})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(nfes, times, "s-", color="tab:red")
    ax2.set_xscale("log")
    ax2.set_xticks(nfes)
    ax2.set_xticklabels([str(x) for x in nfes])
    ax2.set_xlabel("NFE (sampling steps)")
    ax2.set_ylabel("wall-clock time (s)")
    ax2.set_title(f"Speed vs NFE ({head})")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def parse_nfes(raw: str | None) -> list[int]:
    """Parse a comma-separated NFE list (e.g. "1,2,5,10"); sorted + de-duplicated. Default DEFAULT_NFES."""
    if not raw:
        return list(DEFAULT_NFES)
    vals = sorted({int(x) for x in raw.split(",") if x.strip()})
    if not vals or any(v < 1 for v in vals):
        raise SystemExit(f"--nfes must be positive integers, got: {raw!r}")
    return vals


# --------------------------------------------------------------------------- #
# smoke: random tiny models + small NFEs, no checkpoint / no data download
# --------------------------------------------------------------------------- #
def smoke_test() -> None:
    print("=== nfe sweep smoke test ===")
    set_seed(42)
    device = torch.device("cpu")
    n, size = 4, 32
    images = torch.rand(n, 3, size, size, device=device) * 2 - 1
    nfes = [1, 2, 4]
    print(f"images: {tuple(images.shape)}  NFEs: {nfes}")

    for head_kind in ("flow", "diffusion"):
        print(f"\n-- head={head_kind} --")
        model = PatchReconstructionModel(
            img_size=size, patch_size=16, channels=3, dim=32, depth=1, num_heads=2,
            head_kind=head_kind, timesteps=20, head_hidden=32, head_blocks=1, flow_sample_steps=4,
        ).to(device).eval()

        rows = run_sweep(model, images, nfes, runs=2, device=device, seed=42)

        assert len(rows) == len(nfes), "expected one row per NFE"
        assert [r["NFE"] for r in rows] == nfes, "NFE order/identity mismatch"
        for r in rows:
            for key in ("PSNR", "SSIM", "LPIPS", "time_s", "ms_per_img"):
                assert key in r, f"row missing {key}"
            assert r["time_s"] >= 0.0, "negative time"
        # reconstruction must be a valid image batch in range
        recon = reconstruct_nfe(model, images, nfes[-1])
        assert recon.shape == images.shape, recon.shape
        assert torch.isfinite(recon).all() and recon.min() >= -1.0 and recon.max() <= 1.0
        print(format_table(rows))

    print("\nsmoke test PASSED")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep NFE (sampling steps) for a trained PatchReconstructionModel")
    p.add_argument("--smoke", action="store_true", help="CPU smoke: random tiny model + small NFEs, no checkpoint")
    p.add_argument("--ckpt", type=str, default=None, help="path to a trained PatchReconstructionModel checkpoint (.pt)")
    p.add_argument("--head", choices=["flow", "diffusion"], default="flow", help="which head the checkpoint has")
    p.add_argument("--dataset", choices=["cifar10", "ffhq64"], default="cifar10")
    p.add_argument("--data-root", type=str, default="data", help="root for CIFAR-10")
    p.add_argument("--ffhq-root", type=str, default="data/ffhq64", help="folder of FFHQ-64 PNGs")
    p.add_argument("--nfes", type=str, default=None, help='comma-separated NFEs, e.g. "1,2,5,10,20,50" (default)')
    p.add_argument("--eval-images", type=int, default=64, help="size of the fixed test subset")
    p.add_argument("--batch-size", type=int, default=128, help="loader batch size while gathering the subset")
    p.add_argument("--timing-runs", type=int, default=3, help="timed runs averaged per NFE (after 1 warmup)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true", help="also save quality-NFE and time-NFE curves (needs matplotlib)")
    # architecture (must match the checkpoint; defaults match train_patch.py)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--dim", type=int, default=128, help="transformer token dimension D")
    p.add_argument("--depth", type=int, default=4, help="number of transformer blocks")
    p.add_argument("--num-heads", type=int, default=4, help="attention heads")
    p.add_argument("--head-hidden", type=int, default=256, help="generative head MLP width")
    p.add_argument("--head-blocks", type=int, default=3, help="generative head residual blocks")
    p.add_argument("--timesteps", type=int, default=1000, help="diffusion head schedule length T (DDIM respaces within this)")
    p.add_argument("--flow-steps", type=int, default=50, help="flow head construction default (overridden per-NFE in the sweep)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke_test()
        return

    if not args.ckpt:
        raise SystemExit("nothing to sweep: pass --ckpt PATH (a trained PatchReconstructionModel), or --smoke")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    set_seed(args.seed)
    device = get_device()
    image_size = IMAGE_SIZE[args.dataset]
    nfes = parse_nfes(args.nfes)
    print(f"device: {device}  head: {args.head}  dataset: {args.dataset}  NFEs: {nfes}")

    model = build_model(args, image_size, device, args.head)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {ckpt_path}  ({n_params:,} params)")

    images = load_eval_images(args, device, args.eval_images)
    print(f"fixed eval subset: {tuple(images.shape)}\n")

    rows = run_sweep(model, images, nfes, args.timing_runs, device, args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EXPERIMENTS / f"{timestamp}_nfe_{args.head}_{args.dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    md_path = run_dir / "nfe_sweep.md"
    md_path.write_text(
        build_markdown(rows, args.head, args.dataset, str(ckpt_path), images.shape[0],
                       args.timing_runs, device),
        encoding="utf-8",
    )

    print("\n" + format_table(rows))
    print(f"\nMarkdown: {md_path}")
    if args.plot:
        plot_path = maybe_plot(rows, run_dir / "nfe_sweep.png", args.head)
        if plot_path:
            print(f"Plot:     {plot_path}")


if __name__ == "__main__":
    main()
