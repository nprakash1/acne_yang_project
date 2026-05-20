"""
Create classification patches from ACNE04.

For each image we crop:
    - Positive patches: each GT bbox at multiple zoom levels (expand_scales),
      so the network sees acne at the same "lesion area fraction" as DermNet.
    - Negative patches: sampled to be DIVERSE. The default sampler is
      Sobel-energy weighted (texture-biased), which automatically picks
      noses, lips, eyebrows, hair, stubble instead of just smooth cheek.
      Set neg_sampling='uniform' to recover the old behavior.

Output structure:
    out_root/
        train/pos/*.jpg   train/neg/*.jpg
        val/pos/*.jpg     val/neg/*.jpg
        test/pos/*.jpg    test/neg/*.jpg

The split mirrors the underlying ACNE04 train/valid/test splits when present.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # noqa: N816


# ---------------------------------------------------------------------------
# bbox utilities
# ---------------------------------------------------------------------------

def _iou_xyxy(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def _expand_box(box_xywh, scale, W, H):
    x, y, w, h = box_xywh
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * scale
    nx1 = max(0, int(cx - side / 2))
    ny1 = max(0, int(cy - side / 2))
    nx2 = min(W, int(cx + side / 2))
    ny2 = min(H, int(cy + side / 2))
    return [nx1, ny1, nx2, ny2]


# ---------------------------------------------------------------------------
# negative samplers
# ---------------------------------------------------------------------------

def _sample_negative_uniform(gt_xyxy, W, H, side, rng, max_tries=30):
    """Old behavior: random crop with IoU=0 vs every GT box."""
    for _ in range(max_tries):
        if W <= side + 1 or H <= side + 1:
            return None
        x1 = rng.randint(0, W - side)
        y1 = rng.randint(0, H - side)
        cand = [x1, y1, x1 + side, y1 + side]
        if all(_iou_xyxy(cand, g) == 0 for g in gt_xyxy):
            return cand
    return None


def _texture_energy(image_rgb: np.ndarray) -> np.ndarray:
    """Sobel of grayscale + Sobel of HSV saturation.

    The grayscale term fires on luminance edges (eyebrows, nostrils, hair).
    The saturation term fires on chroma edges (lips, blush) that have a
    weak luminance edge. The sum is a robust "interesting face stuff" map.
    """
    if cv2 is None:  # pragma: no cover
        # Without cv2 we fall back to a numpy gradient.
        gray = image_rgb.mean(axis=2).astype(np.float32)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        return np.hypot(gx, gy).astype(np.float32)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_lum = np.hypot(gx, gy)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    sx = cv2.Sobel(sat, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(sat, cv2.CV_32F, 0, 1, ksize=3)
    edge_sat = np.hypot(sx, sy)
    return (edge_lum + 0.5 * edge_sat).astype(np.float32)


def _sample_negatives_textured(
    image_rgb: np.ndarray,
    gt_xyxy: Sequence[Sequence[float]],
    side: int,
    n: int,
    np_rng: np.random.Generator,
) -> list[list[int]]:
    """Sample n negative crops with probability proportional to local Sobel energy.

    Boxes are square of size `side`. Returns xyxy lists. May return fewer than n
    if there is insufficient valid area or the energy map is empty.
    """
    H, W = image_rgb.shape[:2]
    if W <= side + 1 or H <= side + 1:
        return []
    energy = _texture_energy(image_rgb)
    # Zero out around each GT box with a small padding so we never partially
    # overlap a positive.
    pad = side // 2
    for box in gt_xyxy:
        x1, y1, x2, y2 = (int(v) for v in box)
        energy[
            max(0, y1 - pad) : min(H, y2 + pad),
            max(0, x1 - pad) : min(W, x2 + pad),
        ] = 0.0
    # Score each candidate top-left corner by the sum of energy inside its box.
    if cv2 is None:  # pragma: no cover
        pooled = energy
    else:
        pooled = cv2.boxFilter(
            energy, ddepth=cv2.CV_32F, ksize=(side, side), normalize=False
        )
    # Only top-left positions where a full side x side crop fits.
    valid = pooled[: H - side, : W - side]
    flat = valid.ravel().astype(np.float64)
    total = flat.sum()
    if total <= 0:
        return []
    probs = flat / total
    n_pos = int((probs > 0).sum())
    if n_pos == 0:
        return []
    n_draw = min(n, n_pos)
    idx = np_rng.choice(flat.size, size=n_draw, replace=False, p=probs)
    Hv = valid.shape[0]
    Wv = valid.shape[1]
    out: list[list[int]] = []
    for i in idx:
        y1 = int(i // Wv)
        x1 = int(i % Wv)
        # Final defensive IoU=0 check (should already be guaranteed by mask).
        cand = [x1, y1, x1 + side, y1 + side]
        if all(_iou_xyxy(cand, g) == 0 for g in gt_xyxy):
            out.append(cand)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def make_patches(
    coco_root: str,
    out_root: str,
    splits: dict[str, str] | None = None,
    expand: float | None = None,
    expand_scales: Iterable[float] = (1.5, 2.5, 4.0),
    target_size: int = 224,
    neg_per_pos: float = 1.0,
    neg_sampling: str = "texture",  # "texture" | "uniform"
    seed: int = 0,
):
    """Build classification dataset from ACNE04 (COCO-format) detection annotations.

    Parameters
    ----------
    coco_root      : root containing train/, valid/, test/ subdirs with _annotations.coco.json
    out_root       : where to write train/{pos,neg}/, val/{pos,neg}/, test/{pos,neg}/
    splits         : map of dataset-split -> source COCO subdir.
                     default: train/valid/test when test exists
    expand         : if given, override expand_scales with a single legacy scale
    expand_scales  : list of context expansion factors for positives. Default
                     (1.5, 2.5, 4.0) emits one close-up, one medium, one wide
                     crop per GT box so the model sees acne at the same
                     "lesion area fraction" as DermNet at test time.
    target_size    : side length of saved patch (after resize)
    neg_per_pos    : how many negatives per positive PER SCALE
    neg_sampling   : "texture" -> Sobel-energy-weighted (diverse face parts),
                     "uniform" -> legacy IoU=0 random crops (smooth skin only)
    """
    if expand is not None:
        expand_scales = [float(expand)]
    expand_scales = list(expand_scales)
    if neg_sampling not in ("texture", "uniform"):
        raise ValueError(f"neg_sampling must be 'texture' or 'uniform', got {neg_sampling}")

    if splits is None:
        splits = {"train": "train", "val": "valid"}
        if (Path(coco_root) / "test" / "_annotations.coco.json").exists():
            splits["test"] = "test"
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    out_root = Path(out_root)

    counts: dict[str, dict[str, int]] = {}
    for out_split, src_split in splits.items():
        ann_path = Path(coco_root) / src_split / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        anns_by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        n_pos, n_neg = 0, 0
        pos_dir = out_root / out_split / "pos"
        neg_dir = out_root / out_split / "neg"
        pos_dir.mkdir(parents=True, exist_ok=True)
        neg_dir.mkdir(parents=True, exist_ok=True)

        for img in coco["images"]:
            img_path = Path(coco_root) / src_split / img["file_name"]
            if not img_path.exists():
                continue
            pil = Image.open(img_path).convert("RGB")
            W, H = pil.size
            img_np = np.array(pil) if neg_sampling == "texture" else None

            anns = anns_by_img.get(img["id"], [])
            valid_boxes_xywh = []
            gt_xyxy = []
            for a in anns:
                x, y, w, h = a["bbox"]
                if w <= 1 or h <= 1:
                    continue
                valid_boxes_xywh.append([x, y, w, h])
                gt_xyxy.append([x, y, x + w, y + h])

            if not valid_boxes_xywh:
                continue

            stem = Path(img["file_name"]).stem

            # ------------------- POSITIVES (multi-scale) -------------------
            saved_per_scale: dict[float, int] = {}
            scale_sides: dict[float, list[int]] = {}
            for scale in expand_scales:
                for bi, box_xywh in enumerate(valid_boxes_xywh):
                    ex = _expand_box(box_xywh, scale, W, H)
                    side_w = ex[2] - ex[0]
                    side_h = ex[3] - ex[1]
                    if side_w < 8 or side_h < 8:
                        continue
                    crop = pil.crop(ex).resize(
                        (target_size, target_size), Image.BILINEAR
                    )
                    crop.save(
                        pos_dir / f"{stem}_p{bi}_s{scale:.1f}.jpg", quality=92
                    )
                    n_pos += 1
                    saved_per_scale[scale] = saved_per_scale.get(scale, 0) + 1
                    scale_sides.setdefault(scale, []).append(max(side_w, side_h))

            # ------------------- NEGATIVES (per-scale, diverse) -------------
            for scale, n_pos_at_scale in saved_per_scale.items():
                if n_pos_at_scale == 0:
                    continue
                sides_here = scale_sides[scale]
                base = int(np.median(sides_here))
                target_neg = max(1, int(round(n_pos_at_scale * neg_per_pos)))

                if neg_sampling == "texture":
                    # Slight jitter so the negatives don't all have identical side.
                    jitters = np_rng.uniform(0.85, 1.2, size=target_neg)
                    sides = [
                        max(16, min(W, H, int(base * j))) for j in jitters
                    ]
                    # Group requests by side length so we do one Sobel pass per side.
                    by_side: dict[int, int] = {}
                    for s in sides:
                        by_side[s] = by_side.get(s, 0) + 1
                    saved_this_scale = 0
                    for s, n_request in by_side.items():
                        cands = _sample_negatives_textured(
                            img_np, gt_xyxy, s, n_request, np_rng
                        )
                        for cand in cands:
                            crop = pil.crop(cand).resize(
                                (target_size, target_size), Image.BILINEAR
                            )
                            crop.save(
                                neg_dir
                                / f"{stem}_n{n_neg}_s{scale:.1f}.jpg",
                                quality=92,
                            )
                            n_neg += 1
                            saved_this_scale += 1
                else:
                    # uniform legacy sampler
                    for _ in range(target_neg):
                        s = max(16, int(base * py_rng.uniform(0.8, 1.4)))
                        cand = _sample_negative_uniform(
                            gt_xyxy, W, H, s, py_rng
                        )
                        if cand is None:
                            continue
                        crop = pil.crop(cand).resize(
                            (target_size, target_size), Image.BILINEAR
                        )
                        crop.save(
                            neg_dir / f"{stem}_n{n_neg}_s{scale:.1f}.jpg",
                            quality=92,
                        )
                        n_neg += 1

        counts[out_split] = {"pos": n_pos, "neg": n_neg}
        print(f"[{out_split}] pos={n_pos}  neg={n_neg}")

    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--expand-scales",
        type=float,
        nargs="+",
        default=[1.5, 2.5, 4.0],
        help="Context expansion factors for positives (multi-scale).",
    )
    p.add_argument(
        "--expand",
        type=float,
        default=None,
        help="Legacy single-scale expand. If given, overrides --expand-scales.",
    )
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--neg-per-pos", type=float, default=1.0)
    p.add_argument(
        "--neg-sampling",
        default="texture",
        choices=["texture", "uniform"],
        help="texture = Sobel-energy-weighted diverse negatives (default); "
        "uniform = legacy random IoU=0 crops.",
    )
    args = p.parse_args()
    make_patches(
        coco_root=args.coco_root,
        out_root=args.out,
        expand=args.expand,
        expand_scales=args.expand_scales,
        target_size=args.target_size,
        neg_per_pos=args.neg_per_pos,
        neg_sampling=args.neg_sampling,
    )
