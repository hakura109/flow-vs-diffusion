"""Download FFHQ-64 images from the HuggingFace dataset and save them as PNGs.

Pulls from "Dmini/FFHQ-64x64" (streamed, so it does not download the whole 70k set when you
only want a few thousand) and writes zero-padded PNGs into data/ffhq64/, ready for
get_ffhq64_dataloader. Re-running is safe: any image whose target file already exists is skipped.

Usage:
    python scripts/download_ffhq64.py                 # 10000 images -> data/ffhq64/
    python scripts/download_ffhq64.py --num 2000      # fewer images
    python scripts/download_ffhq64.py --out data/ffhq64 --num 10000
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download FFHQ-64 PNGs from HuggingFace")
    p.add_argument("--num", type=int, default=10000, help="number of images to save")
    p.add_argument("--out", type=str, default="data/ffhq64", help="output folder")
    p.add_argument("--repo", type=str, default="Dmini/FFHQ-64x64", help="HuggingFace dataset id")
    p.add_argument("--split", type=str, default="train", help="dataset split to read")
    return p.parse_args()


def main() -> None:
    from datasets import load_dataset
    from PIL import Image as PILImage

    try:
        from tqdm import tqdm
    except ImportError:  # tqdm is optional; fall back to a no-op wrapper
        def tqdm(iterable=None, **kwargs):
            return iterable

    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Zero-pad filenames so they sort correctly (e.g. 00000.png ... 09999.png for --num 10000).
    width = max(5, len(str(args.num - 1)))

    # Stream the dataset so we only fetch what we need instead of the entire archive.
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    img_col: str | None = None
    saved = 0
    skipped = 0
    pbar = tqdm(total=args.num, desc="FFHQ-64")
    for idx, example in enumerate(ds):
        if idx >= args.num:
            break

        path = out / f"{idx:0{width}d}.png"
        if path.exists():
            skipped += 1
            if pbar is not None:
                pbar.update(1)
            continue

        # Detect the PIL-image column once (datasets decode images to PIL by default).
        if img_col is None:
            img_col = next(
                (k for k, v in example.items() if isinstance(v, PILImage.Image)), None
            )
            if img_col is None:
                raise RuntimeError(
                    f"No image column found in {args.repo}; columns are {list(example.keys())}"
                )

        example[img_col].convert("RGB").save(path)
        saved += 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None and hasattr(pbar, "close"):
        pbar.close()

    print(f"Done. Saved {saved} new image(s), skipped {skipped} existing, into {out}/")


if __name__ == "__main__":
    main()
