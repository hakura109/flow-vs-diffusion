"""Image reconstruction metrics: PSNR / SSIM / LPIPS.

All functions expect inputs as tensors (B, C, H, W) in the [-1, 1] range;
internally they convert to whatever range each metric requires.
"""
from __future__ import annotations

from functools import lru_cache

import torch

from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def _to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]."""
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio, higher is better. Inputs [-1, 1], computed in [0, 1] (data_range=1.0)."""
    return peak_signal_noise_ratio(
        _to_unit_range(pred), _to_unit_range(target), data_range=1.0
    )


@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Structural similarity, higher is better (max 1.0). Inputs [-1, 1]."""
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
    """Perceptual distance, lower is better. LPIPS expects inputs in [-1, 1], which matches this project's convention.

    Returns the mean over the batch (a scalar tensor).
    """
    model = _get_lpips_model(net, str(pred.device))
    d = model(pred, target)
    return d.mean()
