"""
Create classification patches from ACNE04.

For each image we crop:
    - Positive patches: each GT bbox expanded by 1.3x context, resized to 224.
    - Negative patches: random crops of the same size distribution that have
      IoU == 0 with every GT bbox in that image.

Output structure:
    out_root/
        train/pos/*.jpg   train/neg/*.jpg
        val/pos/*.jpg     val/neg/*.jpg

The split mirrors the underlying ACNE04 train/valid splits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


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


def _sample_negative(gt_xyxy, W, H, side, rng, max_tries=30):
    for _ in range(max_tries):
        if W <= side + 1 or H <= side + 1:
            return None
        x1 = rng.randint(0, W - side)
        y1 = rng.randint(0, H - side)
        cand = [x1, y1, x1 + side, y1 + side]
        if all(_iou_xyxy(cand, g) == 0 for g in gt_xyxy):
            return cand
    return None


def make_patches(
    coco_root: str,
    out_root: str,
    splits: dict[str, str] | None = None,
    expand: float = 1.3,
    target_size: int = 224,
    neg_per_pos: float = 1.0,
    seed: int = 0,
):
    """Build classification dataset from ACNE04 (COCO-format) detection annotations.

    Parameters
    ----------
    coco_root  : root containing train/, valid/ subdirs each with _annotations.coco.json
    out_root   : where to write train/{pos,neg}/, val/{pos,neg}/
    splits     : map of dataset-split -> source COCO subdir.
                 default: {"train": "train", "val": "valid"}
    expand     : context expansion factor for positives
    target_size: side length of saved patch (after resize)
    neg_per_pos: how many negatives per positive (per image)
    """
    splits = splits or {"train": "train", "val": "valid"}
    rng = random.Random(seed)
    out_root = Path(out_root)

    counts = {}
    for out_split, src_split in splits.items():
        ann_path = Path(coco_root) / src_split / "_annotations.coco.json"
        with open(ann_path) as f:
            coco = json.load(f)

        anns_by_img = {}
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

            anns = anns_by_img.get(img["id"], [])
            gt_xyxy = []
            for a in anns:
                x, y, w, h = a["bbox"]
                if w <= 1 or h <= 1:
                    continue
                # Positive
                ex = _expand_box([x, y, w, h], expand, W, H)
                if ex[2] - ex[0] < 8 or ex[3] - ex[1] < 8:
                    continue
                gt_xyxy.append([x, y, x + w, y + h])
                crop = pil.crop(ex).resize((target_size, target_size), Image.BILINEAR)
                crop.save(pos_dir / f"{Path(img['file_name']).stem}_p{n_pos}.jpg", quality=92)
                n_pos += 1

            # Negatives
            n_neg_target = max(1, int(round(len(gt_xyxy) * neg_per_pos)))
            for k in range(n_neg_target):
                # Choose a side length similar to the lesions (median GT side, with jitter)
                if gt_xyxy:
                    sides = [max(b[2] - b[0], b[3] - b[1]) for b in gt_xyxy]
                    base = int(np.median(sides) * expand)
                else:
                    base = 64
                side = max(16, int(base * rng.uniform(0.8, 1.4)))
                cand = _sample_negative(gt_xyxy, W, H, side, rng)
                if cand is None:
                    continue
                crop = pil.crop(cand).resize((target_size, target_size), Image.BILINEAR)
                crop.save(neg_dir / f"{Path(img['file_name']).stem}_n{n_neg}.jpg", quality=92)
                n_neg += 1

        counts[out_split] = {"pos": n_pos, "neg": n_neg}
        print(f"[{out_split}] pos={n_pos}  neg={n_neg}")

    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expand", type=float, default=1.3)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--neg-per-pos", type=float, default=1.0)
    args = p.parse_args()
    make_patches(
        coco_root=args.coco_root,
        out_root=args.out,
        expand=args.expand,
        target_size=args.target_size,
        neg_per_pos=args.neg_per_pos,
    )
