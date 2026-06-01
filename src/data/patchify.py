"""Image patchify / unpatchify, built on einops.

Tensor layout convention:
  image : (B, C, H, W)
  patch : (B, N, C * p * p), where N = (H // p) * (W // p), unrolled in row-major order.
"""
from __future__ import annotations

import torch
from einops import rearrange


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C*p*p). Requires H and W to be divisible by patch_size."""
    b, c, h, w = images.shape
    p = patch_size
    if h % p != 0 or w % p != 0:
        raise ValueError(f"H={h}, W={w} must be divisible by patch_size={p}")
    return rearrange(
        images, "b c (hp ph) (wp pw) -> b (hp wp) (c ph pw)", ph=p, pw=p
    )


def unpatchify(
    patches: torch.Tensor, patch_size: int, height: int, width: int, channels: int = 3
) -> torch.Tensor:
    """(B, N, C*p*p) -> (B, C, H, W), the inverse of patchify."""
    p = patch_size
    hp = height // p
    wp = width // p
    return rearrange(
        patches,
        "b (hp wp) (c ph pw) -> b c (hp ph) (wp pw)",
        hp=hp,
        wp=wp,
        ph=p,
        pw=p,
        c=channels,
    )


def _roundtrip_test() -> None:
    """patchify -> unpatchify should reconstruct the original image exactly."""
    torch.manual_seed(0)
    b, c, h, w, p = 4, 3, 32, 32, 8
    x = torch.randn(b, c, h, w)
    patches = patchify(x, p)
    expected_n = (h // p) * (w // p)
    assert patches.shape == (b, expected_n, c * p * p), patches.shape
    recon = unpatchify(patches, p, h, w, c)
    assert recon.shape == x.shape, recon.shape
    assert torch.allclose(recon, x, atol=1e-6), "roundtrip mismatch!"
    print(f"[patchify] roundtrip OK: {tuple(x.shape)} -> {tuple(patches.shape)} -> {tuple(recon.shape)}")


if __name__ == "__main__":
    _roundtrip_test()
