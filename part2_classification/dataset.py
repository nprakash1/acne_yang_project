"""
Dataset + albumentations augmentation pipelines for the binary acne classifier.

Augmentation choices are tuned for the ACNE04 -> DermNet domain shift:
    - heavy photometric distortion to mimic dermatology lighting/cameras
    - Gaussian blur + JPEG compression to mimic clinical low-quality images
    - small geometric jitter only (acne shouldn't morph)
    - color-shortcut breakers (ToGray + ChannelShuffle + RGBShift) so the
      model cannot rely on "red = acne" to classify
"""

from __future__ import annotations

import os
import random
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


def get_train_transforms(
    img_size: int = 224,
    heavy_da: bool = True,
    color_shortcut_break: bool = True,
):
    """Strong augmentations for ACNE04 -> DermNet generalization.

    Parameters
    ----------
    heavy_da
        Brightness/contrast/gamma/HSV + blur/JPEG + coarse dropout.
    color_shortcut_break
        Adds ToGray, ChannelShuffle and a wider RGBShift on top of `heavy_da`.
        Designed to prevent the classifier from learning "red blob => acne",
        which is the single biggest cross-domain failure mode on DermNet.
    """
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
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=25, p=1.0),
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
        ]
    if color_shortcut_break:
        # These three together break the "red = acne" shortcut without
        # destroying useful color cues entirely.
        base += [
            A.ToGray(p=0.2),
            A.ChannelShuffle(p=0.15),
            A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.5),
        ]
    if heavy_da:
        base += [
            A.CoarseDropout(max_holes=6, max_height=24, max_width=24, p=0.5),
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


def make_stochastic_preprocess(
    fn: Callable[[np.ndarray], np.ndarray] | None,
    p: float = 0.5,
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Wrap a uint8-RGB->uint8-RGB preprocessor so it only fires with prob `p`.

    Used during training to apply DermNet-style color normalization (Reinhard /
    histogram match) to a random fraction of ACNE04 crops. The model then
    sees source labels but partially-target pixel statistics, which directly
    closes the train/test distribution gap.

    Set `p=1.0` for "always apply" (e.g. test-time).
    """
    if fn is None or p <= 0.0:
        return None
    if p >= 1.0:
        return fn

    def _stochastic(img: np.ndarray) -> np.ndarray:
        if random.random() < p:
            return fn(img)
        return img

    return _stochastic


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
