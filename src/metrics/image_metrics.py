"""图像重建评测指标：PSNR / SSIM / LPIPS。

所有函数的输入约定为 [-1, 1] 范围的张量 (B, C, H, W)；
内部会按各指标的需求转换到合适的范围。
"""
from __future__ import annotations

from functools import lru_cache

import torch

from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def _to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]。"""
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """峰值信噪比，越高越好。输入 [-1, 1]，在 [0, 1] 范围内计算（data_range=1.0）。"""
    return peak_signal_noise_ratio(
        _to_unit_range(pred), _to_unit_range(target), data_range=1.0
    )


@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """结构相似性，越高越好（最大 1.0）。输入 [-1, 1]。"""
    return structural_similarity_index_measure(
        _to_unit_range(pred), _to_unit_range(target), data_range=1.0
    )


@lru_cache(maxsize=2)
def _get_lpips_model(net: str, device: str):
    import lpips as _lpips

    model = _lpips.LPIPS(net=net)
    model.eval()
    return model.to(device)


@torch.no_grad()
def lpips(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> torch.Tensor:
    """感知距离，越低越好。LPIPS 期望输入在 [-1, 1]，正好与本项目约定一致。

    返回 batch 上的平均值（标量张量）。
    """
    model = _get_lpips_model(net, str(pred.device))
    d = model(pred, target)
    return d.mean()
