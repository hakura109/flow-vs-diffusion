"""自编码器训练入口。

三种模式：
    python scripts/train_ae.py --smoke      # CPU 冒烟：2 张合成图，一次前向+反向
    python scripts/train_ae.py --overfit    # 取一个真实 batch 反复训练，验证 loss 能降到 ~0
    python scripts/train_ae.py              # 在真实 CIFAR-10 上完整训练（首次自动下载数据）

完整训练会把 loss 写到 TensorBoard，训练后在测试集上保存重建网格图并计算
PSNR/SSIM/LPIPS，结果统一落在 experiments/<时间戳>_<模式>/ 下。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# 让脚本能直接 import 到项目里的 src 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import denormalize, get_cifar10_dataloader
from src.metrics.image_metrics import lpips, psnr, ssim
from src.models.autoencoder import ConvAutoencoder
from src.utils.setup import get_device, set_seed

EXPERIMENTS = ROOT / "experiments"


def make_run_dir(mode: str, timestamp: str) -> Path:
    run_dir = EXPERIMENTS / f"{timestamp}_{mode}"
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    return run_dir


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# overfit：单 batch 反复训练，验证模型确实能学
# --------------------------------------------------------------------------- #
def overfit(args: argparse.Namespace) -> None:
    print("=== AE overfit (单 batch 反复训练) ===")
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = make_run_dir("overfit", timestamp)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    # 取一个真实 batch（固定不变）
    loader = get_cifar10_dataloader(
        root=args.data_root, train=True, batch_size=args.overfit_batch, shuffle=True
    )
    images, _ = next(iter(loader))
    images = images.to(device)
    print(f"overfit on {images.shape[0]} 张真实图，共 {args.overfit_steps} 步")

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    first_loss = None
    last_loss = None
    for step in range(args.overfit_steps):
        optimizer.zero_grad()
        recon = model(images)
        loss = criterion(recon, images)
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss
        writer.add_scalar("overfit/loss", last_loss, step)
        if step % 50 == 0 or step == args.overfit_steps - 1:
            print(f"  step {step:4d}/{args.overfit_steps}  loss={last_loss:.6f}")

    writer.close()
    print(f"first loss = {first_loss:.6f}  ->  last loss = {last_loss:.6f}")

    assert torch.isfinite(torch.tensor(last_loss)), "loss 不是有限值"
    assert last_loss < first_loss * 0.1, "loss 没有明显下降（<10% 初始值），模型可能没在学"
    if last_loss < 0.01:
        print(f"overfit PASSED ✅  loss 已降到接近 0（{last_loss:.6f}）")
    else:
        print(f"overfit OK ⚠️  loss 大幅下降但未到 0.01（{last_loss:.6f}），可增加步数")
    print(f"TensorBoard 日志: {run_dir / 'tb'}")


# --------------------------------------------------------------------------- #
# 评估 + 可视化
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: nn.Module, loader, device, max_images: int):
    """在测试集（前 max_images 张）上计算 PSNR/SSIM/LPIPS 平均值。"""
    model.eval()
    tot_psnr = tot_ssim = tot_lpips = 0.0
    n = 0
    for images, _ in loader:
        images = images.to(device)
        recon = model(images)
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
    }


@torch.no_grad()
def save_recon_grid(model: nn.Module, images: torch.Tensor, path: Path, n: int = 8) -> None:
    """上排原图、下排重建，存成一张网格图。"""
    model.eval()
    orig = images[:n]
    recon = model(orig)
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
# 完整训练
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> None:
    print("=== AE 完整训练 (CIFAR-10) ===")
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = make_run_dir("train", timestamp)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    train_loader = get_cifar10_dataloader(
        root=args.data_root, train=True, batch_size=args.batch_size
    )
    test_loader = get_cifar10_dataloader(
        root=args.data_root, train=False, batch_size=args.batch_size
    )

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()

            running += loss.item()
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        epoch_loss = running / len(train_loader)
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={epoch_loss:.6f}")

    # 保存权重
    ckpt_path = run_dir / "autoencoder.pt"
    torch.save(model.state_dict(), ckpt_path)

    # 评估 + 可视化
    metrics = evaluate(model, test_loader, device, max_images=args.eval_images)
    for k in ("PSNR", "SSIM", "LPIPS"):
        writer.add_scalar(f"test/{k}", metrics[k], 0)

    vis_images, _ = next(iter(test_loader))
    grid_path = run_dir / "recon_grid.png"
    save_recon_grid(model, vis_images.to(device), grid_path, n=args.grid_images)

    table = format_metrics_table(metrics)
    (run_dir / "metrics.md").write_text(
        f"# AE 测试集评估\n\n上排原图、下排重建：`recon_grid.png`\n\n{table}\n",
        encoding="utf-8",
    )
    writer.close()

    print("\n测试集评估：")
    print(table)
    print(f"\n重建网格图: {grid_path}")
    print(f"权重:       {ckpt_path}")
    print(f"指标表:     {run_dir / 'metrics.md'}")
    print(f"TensorBoard: {run_dir / 'tb'}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train conv autoencoder on CIFAR-10")
    p.add_argument("--smoke", action="store_true", help="CPU 冒烟测试：2 张图，一次前向+反向")
    p.add_argument("--overfit", action="store_true", help="单 batch 反复训练，验证 loss 能降到 ~0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="data")
    # overfit 相关
    p.add_argument("--overfit-steps", type=int, default=300)
    p.add_argument("--overfit-batch", type=int, default=32)
    # 评估/可视化相关
    p.add_argument("--eval-images", type=int, default=512, help="测试集上参与指标平均的图片数")
    p.add_argument("--grid-images", type=int, default=8, help="重建网格每排图片数")
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
