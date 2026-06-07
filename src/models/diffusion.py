"""Minimal DDPM (Denoising Diffusion Probabilistic Models) for whole 32x32 images.

This is the Phase A stepping-stone: a standard, *unconditional* DDPM that works on full
images (no transformer backbone, no per-patch conditioning yet). The goal is only to get
the diffusion mechanism + training / sampling / eval pipeline running end to end.

Conventions:
  - Images live in [-1, 1], matching the data pipeline.
  - The network is a small time-conditioned U-Net. Given a noisy image x_t and its integer
    timestep t, it predicts the noise epsilon that was added (the standard "epsilon-prediction"
    parameterization from Ho et al., 2020).

Kept intentionally minimal and readable, mirroring the canonical minimal DDPM implementations.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# Noise schedule
# --------------------------------------------------------------------------- #
def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Linearly spaced betas from beta_start to beta_end (the original DDPM schedule).

    beta_t is the variance of the Gaussian noise added at forward step t. The default
    endpoints are tuned for T=1000; with a much smaller T the chain simply adds noise faster.
    Returns a 1-D tensor of shape (timesteps,).
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# --------------------------------------------------------------------------- #
# Time-conditioned U-Net building blocks
# --------------------------------------------------------------------------- #
class SinusoidalTimeEmbedding(nn.Module):
    """Map an integer timestep t to a sinusoidal vector (like Transformer positional encoding).

    Output shape: (B, dim). Low frequencies encode coarse position, high frequencies fine,
    which lets the MLP that follows turn t into a smooth conditioning signal.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        # Geometrically spaced frequencies: freq[i] = exp(-log(10000) * i / (half - 1)).
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class ResBlock(nn.Module):
    """Residual block with two conv layers; the timestep embedding is injected in the middle."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        # Project the (shared) time embedding to this block's channel count, added per-channel.
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # 1x1 conv on the skip path when channel counts differ, so the residual add lines up.
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]  # broadcast (B,C) -> (B,C,1,1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Halve spatial resolution with a stride-2 conv (keeps channel count)."""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Double spatial resolution with a stride-2 transposed conv (keeps channel count)."""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SmallUNet(nn.Module):
    """A compact U-Net for 32x32 images: 32 -> 16 -> 8 -> 16 -> 32 with skip connections.

    forward(x_t, t) -> predicted noise, same shape as x_t. Channel plan with C = base_channels:
      down : C , 2C        (at 32) -> 2C (at 16) -> 2C (at 8, the bottleneck)
      up   : 2C (at 16) -> C (at 32), concatenating the matching down-path features as skips.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_emb_dim: int = 256):
        super().__init__()
        c = base_channels

        # Turn the scalar timestep into a conditioning vector shared by every ResBlock.
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(c),
            nn.Linear(c, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

        # --- down path ---
        self.rb1 = ResBlock(c, c, time_emb_dim)          # 32x32, C        (skip s1)
        self.rb2 = ResBlock(c, 2 * c, time_emb_dim)      # 32x32, 2C       (skip s2)
        self.down1 = Downsample(2 * c)                    # 32 -> 16
        self.rb3 = ResBlock(2 * c, 2 * c, time_emb_dim)  # 16x16, 2C       (skip s3)
        self.down2 = Downsample(2 * c)                    # 16 -> 8

        # --- bottleneck ---
        self.mid1 = ResBlock(2 * c, 2 * c, time_emb_dim)
        self.mid2 = ResBlock(2 * c, 2 * c, time_emb_dim)

        # --- up path (each ResBlock consumes a concatenated skip from the down path) ---
        self.up1 = Upsample(2 * c)                        # 8 -> 16
        self.rb4 = ResBlock(2 * c + 2 * c, 2 * c, time_emb_dim)  # concat s3
        self.up2 = Upsample(2 * c)                        # 16 -> 32
        self.rb5 = ResBlock(2 * c + 2 * c, c, time_emb_dim)      # concat s2
        self.rb6 = ResBlock(c + c, c, time_emb_dim)              # concat s1

        self.out_norm = nn.GroupNorm(8, c)
        self.out_conv = nn.Conv2d(c, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)
        s1 = self.rb1(x, t_emb)                  # 32, C
        s2 = self.rb2(s1, t_emb)                 # 32, 2C
        h = self.down1(s2)                       # 16, 2C
        s3 = self.rb3(h, t_emb)                  # 16, 2C
        h = self.down2(s3)                       # 8, 2C

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        h = self.up1(h)                          # 16, 2C
        h = self.rb4(torch.cat([h, s3], dim=1), t_emb)  # 16, 2C
        h = self.up2(h)                          # 32, 2C
        h = self.rb5(torch.cat([h, s2], dim=1), t_emb)  # 32, C
        h = self.rb6(torch.cat([h, s1], dim=1), t_emb)  # 32, C

        return self.out_conv(F.silu(self.out_norm(h)))


# --------------------------------------------------------------------------- #
# Diffusion process (forward noising + reverse sampling)
# --------------------------------------------------------------------------- #
class Diffusion(nn.Module):
    """Holds the noise schedule and implements forward noising, the training loss, and sampling.

    Registered as an nn.Module so the schedule tensors move with `.to(device)` as buffers.
    The denoising network is passed in explicitly (e.g. `diffusion.loss(model, x0)`), keeping
    the process and the architecture decoupled.
    """

    def __init__(self, timesteps: int = 1000):
        super().__init__()
        self.timesteps = timesteps

        betas = linear_beta_schedule(timesteps)            # (T,)
        alphas = 1.0 - betas                                # alpha_t = 1 - beta_t
        alpha_bars = torch.cumprod(alphas, dim=0)           # alpha_bar_t = prod_{s<=t} alpha_s

        # Precompute the coefficients used by q_sample / sampling and store them as buffers.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Gather per-sample scalars values[t] and reshape to (B,1,1,1) for broadcasting."""
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward diffusion in closed form: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1m_ab * noise

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """Pick a random t per image, noise it, have the model predict that noise, return the MSE."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        predicted_noise = model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample_step(self, model: nn.Module, x: torch.Tensor, t_index: int) -> torch.Tensor:
        """One reverse (denoising) step: given x_t, draw x_{t-1} via DDPM ancestral sampling."""
        b = x.shape[0]
        t = torch.full((b,), t_index, device=x.device, dtype=torch.long)
        beta_t = self._extract(self.betas, t, x.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bars, t, x.shape)
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x.shape)

        # Posterior mean of x_{t-1}, expressed through the model's predicted noise.
        predicted_noise = model(x, t)
        mean = sqrt_recip_alpha * (x - beta_t / sqrt_1m_ab * predicted_noise)

        if t_index == 0:
            return mean  # final step is deterministic (no noise added at t=0)
        noise = torch.randn_like(x)
        # Use the fixed variance sigma_t^2 = beta_t (DDPM's simpler "fixed large" choice).
        return mean + torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample(
        self, model: nn.Module, shape: tuple[int, ...], device: torch.device | None = None
    ) -> torch.Tensor:
        """Generate images from pure noise by running the full reverse chain. Returns ~[-1, 1]."""
        if device is None:
            device = self.betas.device
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            x = self.p_sample_step(model, x, i)
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def reconstruct(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t_start: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Denoising reconstruction: noise x0 up to t_start, then denoise back down to t=0.

        Unlike sample() (which starts from pure noise and has no ground truth), this starts from
        a *partially* noised real image, so the result is paired with x0 and PSNR/SSIM/LPIPS are
        meaningful. This is how we put an unconditional DDPM on the reconstruction task for eval.
        """
        if device is None:
            device = self.betas.device
        b = x0.shape[0]
        t = torch.full((b,), t_start, device=device, dtype=torch.long)
        x = self.q_sample(x0, t)  # corrupt to the chosen noise level
        for i in reversed(range(t_start + 1)):
            x = self.p_sample_step(model, x, i)
        return x.clamp(-1.0, 1.0)
