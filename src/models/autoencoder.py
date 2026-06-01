"""Small convolutional autoencoder for 32x32 image reconstruction.

Both input and output are in the [-1, 1] range (Tanh on the output), consistent
with the data normalization convention.
encoder: 32 -> 16 -> 8 (4x downsampling); decoder upsamples symmetrically back to 32.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, latent_channels: int = 128):
        super().__init__()
        c = base_channels

        self.encoder = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(in_channels, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            # 8x8 -> 8x8 (latent)
            nn.Conv2d(c * 2, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(latent_channels, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            # 32x32 -> 32x32, squashed into [-1, 1]
            nn.Conv2d(c, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
