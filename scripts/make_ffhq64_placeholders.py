"""生成若干张 64x64 占位图，让 FFHQ-64 数据管道在拿到真实数据前就能跑通。

用法：
    python scripts/make_ffhq64_placeholders.py            # 默认 48 张 -> data/ffhq64/
    python scripts/make_ffhq64_placeholders.py --n 100 --out data/ffhq64

生成的是随机渐变 + 椭圆/矩形的合成图，仅用于验证管道；
拿到真实 FFHQ-64 图片后，直接把它们放进同一目录即可（占位图可删）。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def _make_one(size: int, rng: random.Random) -> Image.Image:
    # 竖直渐变背景
    top = tuple(rng.randint(0, 255) for _ in range(3))
    bottom = tuple(rng.randint(0, 255) for _ in range(3))
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        t = y / max(size - 1, 1)
        color = tuple(int(top[c] * (1 - t) + bottom[c] * t) for c in range(3))
        for x in range(size):
            px[x, y] = color

    # 叠几个随机形状，增加多样性
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
    p.add_argument("--n", type=int, default=48, help="生成图片数量")
    p.add_argument("--out", type=str, default="data/ffhq64", help="输出目录")
    p.add_argument("--size", type=int, default=64, help="图片边长")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for i in range(args.n):
        img = _make_one(args.size, rng)
        img.save(out / f"placeholder_{i:04d}.png")

    print(f"已生成 {args.n} 张 {args.size}x{args.size} 占位图 -> {out.resolve()}")
    print("提示：拿到真实 FFHQ-64 后把图片放进同一目录即可（占位图可删）。")


if __name__ == "__main__":
    main()
