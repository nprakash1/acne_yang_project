"""
Pixel-level domain adaptation utilities for ACNE04 -> DermNet.

Implements:
    - histogram_match(src, ref): match each channel of src to the cumulative
      histogram of ref. Operates on np.uint8 RGB.
    - reinhard_normalize(src, ref): match mean+std in LAB color space.
    - build_reference_mosaic(paths): take a few DermNet images and build a
      single reference image that summarizes the target domain statistics.

These are applied **at test time** to DermNet inputs, making them statistically
look more like the ACNE04 training images.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def histogram_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Channel-wise CDF matching. Both images uint8 HxWx3 RGB."""
    matched = np.empty_like(src)
    for c in range(3):
        s = src[..., c].ravel()
        r = ref[..., c].ravel()
        s_vals, s_counts = np.unique(s, return_counts=True)
        r_vals, r_counts = np.unique(r, return_counts=True)
        s_q = np.cumsum(s_counts).astype(np.float64) / s.size
        r_q = np.cumsum(r_counts).astype(np.float64) / r.size
        # For each src value, find closest CDF level in ref
        interp = np.interp(s_q, r_q, r_vals)
        mapping = np.zeros(256, dtype=np.uint8)
        mapping[s_vals] = np.clip(interp, 0, 255).astype(np.uint8)
        # Fill gaps so unseen src values still get mapped
        last = 0
        for v in range(256):
            if mapping[v] == 0 and v not in s_vals:
                mapping[v] = last
            else:
                last = mapping[v]
        matched[..., c] = mapping[src[..., c]]
    return matched


def reinhard_normalize(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match mean+std in LAB color space (Reinhard et al. 2001)."""
    if cv2 is None:
        # Fallback: RGB mean/std match
        s = src.astype(np.float32)
        r = ref.astype(np.float32)
        for c in range(3):
            sm, ss = s[..., c].mean(), s[..., c].std() + 1e-6
            rm, rs = r[..., c].mean(), r[..., c].std() + 1e-6
            s[..., c] = (s[..., c] - sm) * (rs / ss) + rm
        return np.clip(s, 0, 255).astype(np.uint8)

    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = np.empty_like(src_lab)
    for c in range(3):
        sm, ss = src_lab[..., c].mean(), src_lab[..., c].std() + 1e-6
        rm, rs = ref_lab[..., c].mean(), ref_lab[..., c].std() + 1e-6
        out[..., c] = (src_lab[..., c] - sm) * (rs / ss) + rm
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def build_reference_mosaic(
    image_paths: Iterable[str | os.PathLike],
    tile_size: int = 224,
    grid: int = 4,
) -> np.ndarray:
    """Tile up to grid*grid images into a single reference image.

    Used to summarize the target (DermNet) color distribution from the
    20 unlabeled samples allowed by the assignment.
    """
    paths = [Path(p) for p in image_paths]
    paths = paths[: grid * grid]
    canvas = np.zeros((tile_size * grid, tile_size * grid, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        r, c = divmod(i, grid)
        img = np.array(Image.open(p).convert("RGB").resize((tile_size, tile_size)))
        canvas[r * tile_size : (r + 1) * tile_size, c * tile_size : (c + 1) * tile_size] = img
    return canvas


def make_preprocessor(
    reference_image: np.ndarray | None,
    method: str = "reinhard",
):
    """Return a function np.uint8 RGB -> np.uint8 RGB.

    method in {"none","histogram","reinhard"}.
    """
    if reference_image is None or method == "none":
        return None

    ref = reference_image

    def _fn(img: np.ndarray) -> np.ndarray:
        if method == "histogram":
            return histogram_match(img, ref)
        elif method == "reinhard":
            return reinhard_normalize(img, ref)
        else:
            raise ValueError(method)

    return _fn
