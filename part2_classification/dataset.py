"""
Dataset + albumentations augmentation pipelines for the binary acne classifier.

Augmentation choices are tuned for the ACNE04 -> DermNet domain shift:
    - heavy photometric distortion to mimic dermatology lighting/cameras
    - Gaussian blur + JPEG compression to mimic clinical low-quality images
    - small geometric jitter only (acne shouldn't morph)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:  # pragma: no cover
    A = None
    ToTensorV2 = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224, heavy_da: bool = True):
    """Strong augmentations for ACNE04 -> DermNet generalization."""
    if A is None:
        raise RuntimeError("albumentations is required")
    base = [
        A.LongestMaxSize(max_size=int(img_size * 1.15)),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.RandomCrop(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
    ]
    if heavy_da:
        base += [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0),
                ],
                p=0.85,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.ImageCompression(quality_lower=40, quality_upper=85, p=1.0),
                ],
                p=0.5,
            ),
            A.CoarseDropout(max_holes=4, max_height=24, max_width=24, p=0.3),
        ]
    base += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(base)


def get_eval_transforms(img_size: int = 224):
    if A is None:
        raise RuntimeError("albumentations is required")
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class PatchFolder(Dataset):
    """ImageFolder-style dataset for {root}/pos and {root}/neg."""

    CLASSES = ("neg", "pos")

    def __init__(
        self,
        root: str | os.PathLike,
        transform: Callable | None = None,
        preprocess: Callable | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.preprocess = preprocess  # e.g. histogram match -- np.uint8 in/out
        self.samples: list[tuple[Path, int]] = []
        for cls_idx, cls_name in enumerate(self.CLASSES):
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((p, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.preprocess is not None:
            img = self.preprocess(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)


def class_weights(dataset: PatchFolder) -> torch.Tensor:
    counts = np.bincount([s[1] for s in dataset.samples], minlength=2).astype(float)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32)
