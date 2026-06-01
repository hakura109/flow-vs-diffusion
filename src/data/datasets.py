"""Data loading: CIFAR-10 and FFHQ-64, with pixels normalized to [-1, 1]."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".ppm"}

# CIFAR-10 raw pixels are in [0, 1]; the line below maps them linearly to [-1, 1]: x * 2 - 1
_TO_TANH_RANGE = transforms.Compose(
    [
        transforms.ToTensor(),  # -> [0, 1], shape (C, H, W)
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # -> [-1, 1]
    ]
)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Map a [-1, 1] tensor back to [0, 1], handy for visualization or saving images."""
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def get_cifar10_dataset(root: str = "data", train: bool = True, download: bool = True):
    return datasets.CIFAR10(
        root=root, train=train, download=download, transform=_TO_TANH_RANGE
    )


def get_cifar10_dataloader(
    root: str = "data",
    train: bool = True,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool | None = None,
    download: bool = True,
) -> DataLoader:
    """Return a CIFAR-10 DataLoader with images normalized to [-1, 1].

    shuffle defaults to match train (shuffle the training set, not the test set).
    """
    dataset = get_cifar10_dataset(root=root, train=train, download=download)
    if shuffle is None:
        shuffle = train
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=train,
    )


# --------------------------------------------------------------------------- #
# FFHQ-64: read from a local image folder (ImageFolder-style, no class subdirs needed)
# --------------------------------------------------------------------------- #
class FlatImageDataset(Dataset):
    """Recursively read all images under a folder and return (image_tensor, 0).

    Aligned with torchvision's classification dataset interface (each sample is an
    (image, label) pair), but does not require class subdirectories -- unlabeled data
    like FFHQ is usually just a flat pile of images.
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(
                f"Image directory {self.root!s} not found. Put FFHQ-64 images there, "
                f"or first run scripts/make_ffhq64_placeholders.py to generate placeholders."
            )
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in _IMG_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No images found under {self.root!s} (supported extensions {sorted(_IMG_EXTS)}). "
                f"You can run scripts/make_ffhq64_placeholders.py to generate placeholders."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def _ffhq64_transform(image_size: int = 64) -> transforms.Compose:
    # Resize the short side to image_size, center-crop to a square, then normalize to [-1, 1]
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # -> [0, 1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # -> [-1, 1]
        ]
    )


def get_ffhq64_dataset(
    root: str = "data/ffhq64", image_size: int = 64
) -> FlatImageDataset:
    return FlatImageDataset(root=root, transform=_ffhq64_transform(image_size))


def get_ffhq64_dataloader(
    root: str = "data/ffhq64",
    train: bool = True,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool | None = None,
    image_size: int = 64,
) -> DataLoader:
    """Return an FFHQ-64 DataLoader with images resized to 64x64 and normalized to [-1, 1].

    The interface matches get_cifar10_dataloader (FFHQ has no built-in train/test split,
    so train only controls the default behavior of shuffle / drop_last).
    """
    dataset = get_ffhq64_dataset(root=root, image_size=image_size)
    if shuffle is None:
        shuffle = train
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=train,
    )
