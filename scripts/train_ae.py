"""自编码器训练入口。

冒烟测试：
    python scripts/train_ae.py --smoke

冒烟模式只取 2 张图，在 CPU 上做一次前向 + 反向，打印输出形状与 loss，
不下载数据集、不写文件，用来快速确认代码不崩。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

# 让脚本能直接 import 到项目里的 src 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.autoencoder import ConvAutoencoder
from src.utils.setup import get_device, set_seed


def smoke_test() -> None:
    print("=== AE smoke test ===")
    set_seed(42)
    device = torch.device("cpu")  # 冒烟测试强制 CPU
    print(f"device: {device}")

    # 2 张合成的 32x32 RGB 图，范围 [-1, 1]（不依赖数据下载）
    x = torch.rand(2, 3, 32, 32, device=device) * 2 - 1
    print(f"input shape : {tuple(x.shape)}  range=[{x.min():.3f}, {x.max():.3f}]")

    model = ConvAutoencoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 一次前向 + 反向
    model.train()
    optimizer.zero_grad()
    recon = model(x)
    loss = criterion(recon, x)
    loss.backward()
    optimizer.step()

    print(f"output shape: {tuple(recon.shape)}  range=[{recon.min():.3f}, {recon.max():.3f}]")
    print(f"loss        : {loss.item():.6f}")

    assert recon.shape == x.shape, "输出形状与输入不一致"
    assert torch.isfinite(loss), "loss 不是有限值"
    print("smoke test PASSED ✅")


def train(args: argparse.Namespace) -> None:
    """完整训练循环（占位，云端 GPU 上跑）。"""
    from src.data.datasets import get_cifar10_dataloader

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    loader = get_cifar10_dataloader(
        root=args.data_root, train=True, batch_size=args.batch_size
    )
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  loss={running / len(loader):.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train conv autoencoder on CIFAR-10")
    p.add_argument("--smoke", action="store_true", help="CPU 冒烟测试：2 张图，一次前向+反向")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
