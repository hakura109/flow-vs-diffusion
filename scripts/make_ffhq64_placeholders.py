"""Generate a handful of 64x64 placeholder images so the FFHQ-64 data pipeline
can run before real data is available.

Usage:
    python scripts/make_ffhq64_placeholders.py            # 48 images by default -> data/ffhq64/
    python scripts/make_ffhq64_placeholders.py --n 100 --out data/ffhq64

These are synthetic images (random gradients + ellipses/rectangles), only meant to
exercise the pipeline. Once you have real FFHQ-64 images, just drop them into the same
directory (the placeholders can be deleted).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def _make_one(size: int, rng: random.Random) -> Image.Image:
    # Vertical gradient background
    top = tuple(rng.randint(0, 255) for _ in range(3))
    bottom = tuple(rng.randint(0, 255) for _ in range(3))
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        t = y / max(size - 1, 1)
        color = tuple(int(top[c] * (1 - t) + bottom[c] * t) for c in range(3))
        for x in range(size):
            px[x, y] = color

    # Overlay a few random shapes to add variety
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(2, 4)):
        x0, y0 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        x1, y1 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        fill = tuple(rng.randint(0, 255) for _ in range(3))
        if rng.random() < 0.5:
            draw.ellipse(box, fill=fill)
        else:
            draw.rectangle(box, fill=fill)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description="Generate placeholder FFHQ-64 images")
    p.add_argument("--n", type=int, default=48, help="number of images to generate")
    p.add_argument("--out", type=str, default="data/ffhq64", help="output directory")
    p.add_argument("--size", type=int, default=64, help="image side length")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for i in range(args.n):
        img = _make_one(args.size, rng)
        img.save(out / f"placeholder_{i:04d}.png")

    print(f"Generated {args.n} placeholder {args.size}x{args.size} images -> {out.resolve()}")
    print("Tip: once you have real FFHQ-64, drop the images into the same directory (placeholders can be deleted).")


if __name__ == "__main__":
    main()
