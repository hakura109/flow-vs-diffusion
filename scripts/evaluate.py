"""Evaluate and compare trained reconstruction models on one shared batch of test images.

Loads one or more checkpoints, reconstructs the *same* fixed set of test images with each,
and prints PSNR/SSIM/LPIPS. Designed to compare models head-to-head — e.g. later a diffusion
head vs a flow-matching head.

Important: different model kinds solve *different reconstruction tasks*, so their numbers are
NOT directly comparable:
  - ae              : direct reconstruction (encode -> decode the clean image).
  - diffusion       : denoising reconstruction (noise the image to t_start, then denoise back).
  - patch-diffusion : full-noise patch reconstruction (encode an image to per-patch hidden states,
                      then sample each patch from pure noise given its hidden state) -- diffusion head.
  - patch-flow      : the SAME full-noise patch reconstruction, but with a flow-matching head.
patch-diffusion and patch-flow share one protocol/group ("patch-recon"), so they land in the SAME
table and ARE directly comparable -- this is the project's headline, capacity-matched
diffusion-vs-flow comparison. The whole-image ae/diffusion baselines use different bottlenecks and
protocols, so they stay in their own tables for reference only. Results are grouped by protocol and
printed as one table per protocol; rows in different tables must not be compared.

All inputs are kept in the project's [-1, 1] convention; reconstructions are clamped to [-1, 1]
before scoring so every model is measured on the same footing.

Usage:
    # smoke: tiny random subset + randomly-initialized models, no checkpoints needed
    python scripts/evaluate.py --smoke

    # compare a trained AE and a trained DDPM on 64 CIFAR-10 test images
    python scripts/evaluate.py \
        --model ae        experiments/<ts>_cifar10_train/autoencoder.pt   AE \
        --model diffusion experiments/<ts>_diffusion_cifar10_train/diffusion_unet.pt  DDPM \
        --dataset cifar10 --eval-images 64

    # the headline comparison: patch-diffusion vs patch-flow in ONE shared table
    python scripts/evaluate.py \
        --model patch-diffusion experiments/<ts>_patch_diffusion_cifar10_train/patch_diffusion.pt  PatchDiff \
        --model patch-flow      experiments/<ts>_patch_flow_cifar10_train/patch_flow.pt            PatchFlow \
        --dataset cifar10 --eval-images 64
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import torch

# Let the script import the project's src package directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import get_cifar10_dataloader, get_ffhq64_dataloader
from src.metrics.image_metrics import lpips, psnr, ssim
from src.models.autoencoder import ConvAutoencoder
from src.models.diffusion import Diffusion, SmallUNet
from src.models.patch_transformer import PatchReconstructionModel
from src.utils.setup import get_device, set_seed

IMAGE_SIZE = {"cifar10": 32, "ffhq64": 64}


# --------------------------------------------------------------------------- #
# A reconstructor: a named model bound to the task/protocol it is scored under.
# --------------------------------------------------------------------------- #
@dataclass
class Reconstructor:
    label: str                              # display name in the table
    protocol: str                           # human-readable task description (table heading)
    group: str                              # comparability key: only same-group rows share a table
    fn: Callable[[torch.Tensor], torch.Tensor]  # images[-1,1] -> reconstruction[-1,1]


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
    """Load the first n test images in a fixed order (shuffle=False), so every run/model sees the same batch."""
    loader = build_loader(args, train=False, batch_size=args.batch_size, shuffle=False)
    chunks, got = [], 0
    for images, _ in loader:
        chunks.append(images)
        got += images.shape[0]
        if got >= n:
            break
    x = torch.cat(chunks, dim=0)[:n].to(device)
    return x


# --------------------------------------------------------------------------- #
# building reconstructors from checkpoints
# --------------------------------------------------------------------------- #
def build_ae(ckpt: Path, label: str, args, device, image_size) -> Reconstructor:
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    @torch.no_grad()
    def fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    return Reconstructor(label, "Direct reconstruction (AE encode -> decode)", "direct", fn)


def build_diffusion(ckpt: Path, label: str, args, device, image_size) -> Reconstructor:
    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    diffusion = Diffusion(timesteps=args.timesteps).to(device)
    t_start = int(args.recon_t_frac * args.timesteps)

    @torch.no_grad()
    def fn(x: torch.Tensor) -> torch.Tensor:
        return diffusion.reconstruct(model, x, t_start, device=device)

    protocol = f"Denoising reconstruction (noise to t_start={t_start}/{args.timesteps}, then denoise)"
    return Reconstructor(label, protocol, f"denoise@{t_start}/{args.timesteps}", fn)


# Full-noise patch reconstruction: encode an image to per-patch hidden states, then sample every
# patch from PURE noise conditioned on its hidden state. Shared by both patch heads so that
# patch-diffusion and patch-flow are scored under one protocol and compared in a single table.
PATCH_PROTOCOL = "Full-noise patch reconstruction (encode -> sample each patch from noise given its hidden state)"


def build_patch(ckpt: Path, label: str, args, device, image_size, *, head_kind: str) -> Reconstructor:
    """Build a PatchReconstructionModel reconstructor (diffusion or flow head).

    Architecture hyperparameters must match how the checkpoint was trained: pass the same
    --patch-size / --dim / --depth / --num-heads / --head-hidden / --head-blocks the run used
    (defaults match train_patch.py). `head_kind` is fixed by the registry key
    (`patch-diffusion` -> "diffusion", `patch-flow` -> "flow").

    Both heads reconstruct each patch from pure noise conditioned on its per-patch hidden state, so
    they share one protocol/group ("patch-recon"): patch-diffusion and patch-flow land in the SAME
    table and are directly comparable -- the fair, capacity-matched diffusion-vs-flow comparison.
    """
    model = PatchReconstructionModel(
        img_size=image_size,
        patch_size=args.patch_size,
        channels=3,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        head_kind=head_kind,
        timesteps=args.timesteps,           # diffusion head schedule length
        head_hidden=args.head_hidden,
        head_blocks=args.head_blocks,
        flow_sample_steps=args.flow_steps,  # flow head Euler steps
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    @torch.no_grad()
    def fn(x: torch.Tensor) -> torch.Tensor:
        return model.reconstruct(x)

    return Reconstructor(label, PATCH_PROTOCOL, "patch-recon", fn)


# Registry. patch-diffusion / patch-flow both use build_patch with head_kind fixed by the key, so
# they share the "patch-recon" table (directly comparable); ae / diffusion stay in their own tables.
BUILDERS: dict[str, Callable[..., Reconstructor]] = {
    "ae": build_ae,
    "diffusion": build_diffusion,
    "patch-diffusion": partial(build_patch, head_kind="diffusion"),
    "patch-flow": partial(build_patch, head_kind="flow"),
}


def parse_model_specs(specs: list[list[str]], args, device, image_size) -> list[Reconstructor]:
    recs = []
    for spec in specs:
        if len(spec) < 2:
            raise SystemExit(f"--model needs at least KIND and PATH, got: {spec}")
        kind, path = spec[0], spec[1]
        label = spec[2] if len(spec) >= 3 else f"{kind}:{Path(path).parent.name}"
        if kind not in BUILDERS:
            raise SystemExit(f"unknown model kind '{kind}'; choices: {sorted(BUILDERS)}")
        ckpt = Path(path)
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found: {ckpt}")
        recs.append(BUILDERS[kind](ckpt, label, args, device, image_size))
    return recs


# --------------------------------------------------------------------------- #
# metrics + reporting
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


def format_group_table(protocol: str, rows: list[tuple[str, dict]]) -> str:
    lines = [
        f"### {protocol}",
        "",
        "| Model | PSNR (dB) | SSIM | LPIPS |",
        "| ----- | --------- | ---- | ----- |",
    ]
    for label, m in rows:
        lines.append(f"| {label} | {m['PSNR']:.3f} | {m['SSIM']:.4f} | {m['LPIPS']:.4f} |")
    return "\n".join(lines)


def report(recs: list[Reconstructor], images: torch.Tensor, dataset: str) -> str:
    """Score every reconstructor on the shared `images`, grouped into one table per protocol."""
    n = images.shape[0]
    results: dict[str, dict] = {}  # group -> {"protocol":..., "rows":[(label, metrics)]}
    for rec in recs:
        recon = rec.fn(images)
        metrics = compute_metrics(recon, images)
        g = results.setdefault(rec.group, {"protocol": rec.protocol, "rows": []})
        g["rows"].append((rec.label, metrics))
        print(f"  scored {rec.label}: PSNR {metrics['PSNR']:.3f}  "
              f"SSIM {metrics['SSIM']:.4f}  LPIPS {metrics['LPIPS']:.4f}")

    parts = [f"# Reconstruction evaluation\n",
             f"Dataset: **{dataset}** | shared test images: **{n}** | inputs in [-1, 1].\n"]
    if len(results) > 1:
        parts.append("> NOTE: tables below use different reconstruction tasks/protocols. "
                     "Compare rows WITHIN a table only, never across tables.\n")
    for g in results.values():
        parts.append(format_group_table(g["protocol"], g["rows"]))
        parts.append("")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# smoke: random subset + randomly-initialized models (no checkpoints needed)
# --------------------------------------------------------------------------- #
def smoke_test() -> None:
    print("=== evaluate smoke test ===")
    set_seed(42)
    device = torch.device("cpu")
    n, size, T = 4, 32, 20  # tiny everything so the diffusion reverse chain is cheap on CPU
    images = torch.rand(n, 3, size, size, device=device) * 2 - 1
    print(f"images: {tuple(images.shape)}  range=[{images.min():.3f}, {images.max():.3f}]")

    # --- whole-image baselines (random init): each its own protocol/group, reference only ---
    ae = ConvAutoencoder().to(device).eval()
    unet = SmallUNet().to(device).eval()
    diffusion = Diffusion(timesteps=T).to(device)

    @torch.no_grad()
    def ae_fn(x):
        return ae(x)

    @torch.no_grad()
    def diff_fn(x):
        return diffusion.reconstruct(unet, x, t_start=T // 2, device=device)

    recs = [
        Reconstructor("AE(random)", "Direct reconstruction (AE encode -> decode)", "direct", ae_fn),
        Reconstructor("DDPM(random)", f"Denoising reconstruction (t_start={T // 2}/{T})",
                      f"denoise@{T // 2}/{T}", diff_fn),
    ]

    # --- patch heads: exercise the REAL build_patch via the registry with tiny temp checkpoints ---
    # A tiny architecture so the reverse chain / Euler integration is cheap on CPU. build_patch reads
    # these same values off `args`, so the saved and reloaded models match exactly (load_state_dict).
    args = argparse.Namespace(
        patch_size=16, dim=32, depth=1, num_heads=2,
        head_hidden=32, head_blocks=1, timesteps=10, flow_steps=4,
    )
    with tempfile.TemporaryDirectory() as tmp:
        for head_kind, key, lbl in (
            ("diffusion", "patch-diffusion", "patch-diff(random)"),
            ("flow", "patch-flow", "patch-flow(random)"),
        ):
            m = PatchReconstructionModel(
                img_size=size, patch_size=args.patch_size, channels=3,
                dim=args.dim, depth=args.depth, num_heads=args.num_heads,
                head_kind=head_kind, timesteps=args.timesteps,
                head_hidden=args.head_hidden, head_blocks=args.head_blocks,
                flow_sample_steps=args.flow_steps,
            )
            ckpt = Path(tmp) / f"{key}.pt"
            torch.save(m.state_dict(), ckpt)
            recs.append(BUILDERS[key](ckpt, lbl, args, device, size))  # the real builder path

        out = report(recs, images, dataset="synthetic")

    print("\n" + out)

    # Three distinct protocols -> three tables (direct, denoising, patch-recon).
    assert out.count("### ") == 3, "expected three protocol tables (direct, denoising, patch-recon)"
    # The two patch heads must MERGE into one shared table (the headline diffusion-vs-flow comparison).
    assert out.count(PATCH_PROTOCOL) == 1, "patch-diffusion and patch-flow must share one table"
    assert "patch-diff(random)" in out and "patch-flow(random)" in out, "both patch rows must appear"
    print("smoke test PASSED")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate/compare reconstruction models on one shared test batch")
    p.add_argument("--smoke", action="store_true", help="tiny random subset + random-init models, no checkpoints")
    p.add_argument("--model", action="append", nargs="+", default=[], metavar=("KIND PATH", "LABEL"),
                   help="repeatable: KIND(ae|diffusion|patch-diffusion|patch-flow) PATH [LABEL]; "
                        "same protocol -> same table")
    p.add_argument("--dataset", choices=["cifar10", "ffhq64"], default="cifar10")
    p.add_argument("--data-root", type=str, default="data", help="root for CIFAR-10")
    p.add_argument("--ffhq-root", type=str, default="data/ffhq64", help="folder of FFHQ-64 PNGs")
    p.add_argument("--eval-images", type=int, default=64, help="number of shared test images to score on")
    p.add_argument("--batch-size", type=int, default=128, help="loader batch size while gathering the eval images")
    p.add_argument("--seed", type=int, default=42)
    # denoising-reconstruction protocol (whole-image diffusion baseline); --timesteps doubles as the
    # diffusion patch head's schedule length.
    p.add_argument("--timesteps", type=int, default=1000, help="T for the diffusion schedule (diffusion + patch-diffusion)")
    p.add_argument("--recon-t-frac", type=float, default=0.5, help="denoising recon starts at t = frac * T")
    # patch model architecture (must match the trained checkpoint; defaults match train_patch.py)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--dim", type=int, default=128, help="transformer token dimension D")
    p.add_argument("--depth", type=int, default=4, help="number of transformer blocks")
    p.add_argument("--num-heads", type=int, default=4, help="attention heads")
    p.add_argument("--head-hidden", type=int, default=256, help="generative head MLP width")
    p.add_argument("--head-blocks", type=int, default=3, help="generative head residual blocks")
    p.add_argument("--flow-steps", type=int, default=50, help="flow head: number of Euler integration steps")
    p.add_argument("--out", type=str, default=None, help="optional path to also write the markdown report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke_test()
        return

    if not args.model:
        raise SystemExit("nothing to evaluate: pass --model KIND PATH [LABEL] (repeatable), or --smoke")

    set_seed(args.seed)
    device = get_device()
    image_size = IMAGE_SIZE[args.dataset]
    print(f"device: {device}  dataset: {args.dataset}  eval-images: {args.eval_images}")

    images = load_eval_images(args, device, args.eval_images)
    print(f"loaded {images.shape[0]} shared test images {tuple(images.shape[1:])}")

    recs = parse_model_specs(args.model, args, device, image_size)
    out = report(recs, images, args.dataset)
    print("\n" + out)

    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
