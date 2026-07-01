#!/usr/bin/env bash
set -e
for p in 4 8 16; do
  for head in flow diffusion; do
    echo "=== patch-size=$p  head=$head ==="
    python scripts/train_patch.py --dataset ffhq64 --head $head --patch-size $p
  done
done
