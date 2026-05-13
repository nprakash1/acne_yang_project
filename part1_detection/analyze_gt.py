"""GT box-size + density analysis for ACNE04.

Run this once before tuning detectors. Outputs:
  - outputs/eda/gt_size_stats.json           (machine-readable summary)
  - outputs/eda/gt_size_distribution.png     (histogram of sqrt(w*h))
  - outputs/eda/gt_aspect_distribution.png   (histogram of w/h)
  - outputs/eda/gt_per_image.png             (lesions per image)

Print a recommended `anchor_sizes` tuple based on the measured size distribution.

Usage:
    python -m part1_detection.analyze_gt --coco data/acne04/coco/train/_annotations.coco.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


# FPN strides for ResNet-50-FPN. Anchor sizes should track these so each level
# proposes objects of a size that fits its receptive field.
FPN_STRIDES = (4, 8, 16, 32, 64)


def analyze(coco_path: str, out_dir: str = "outputs/eda") -> dict:
    coco_path = Path(coco_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(coco_path.read_text())
    anns = coco["annotations"]
    imgs = coco["images"]
    print(f"Loaded {len(anns):,} annotations across {len(imgs):,} images "
          f"from {coco_path}")

    # ---- box size + aspect ratio ----
    widths = np.array([a["bbox"][2] for a in anns if a["bbox"][2] > 0])
    heights = np.array([a["bbox"][3] for a in anns if a["bbox"][3] > 0])
    sizes = np.sqrt(widths * heights)            # geometric mean (px)
    ratios = widths / heights                    # w/h
    image_w = np.array([i["width"] for i in imgs])
    image_h = np.array([i["height"] for i in imgs])

    # ---- per-image lesion count ----
    per_image = Counter(a["image_id"] for a in anns)
    counts = np.array(list(per_image.values()))

    pct = lambda arr, p: float(np.percentile(arr, p))
    summary = {
        "n_images": len(imgs),
        "n_boxes": int(len(sizes)),
        "image_size_median": [float(np.median(image_w)), float(np.median(image_h))],
        "box_size_px": {f"p{p}": pct(sizes, p) for p in (1, 5, 25, 50, 75, 95, 99)},
        "aspect_ratio": {f"p{p}": pct(ratios, p) for p in (5, 25, 50, 75, 95)},
        "boxes_per_image": {f"p{p}": pct(counts, p) for p in (5, 50, 95)},
        "boxes_per_image_max": int(counts.max()),
    }

    # ---- recommended anchor sizes ----
    # Strategy: pick 5 anchor *base sizes* spanning the box-size distribution
    # using the p5, p25, p50, p75, p95 percentiles. Each level then gets two
    # nearby sizes to add some scale variation.
    pivots = [pct(sizes, p) for p in (5, 25, 50, 75, 95)]
    pivots = [max(2.0, round(p)) for p in pivots]  # never go below 2 px

    # Build a (low, high) tuple per FPN level by spreading +/- ~25% around the pivot.
    rec_anchor_sizes = tuple(
        (max(2, int(round(p * 0.8))), max(3, int(round(p * 1.25))))
        for p in pivots
    )
    # Build recommended aspect ratios from p25/p50/p75 of measured ratios.
    rec_aspect_ratios = tuple(
        round(pct(ratios, p), 2) for p in (25, 50, 75)
    )

    summary["recommended_anchor_sizes"] = rec_anchor_sizes
    summary["recommended_aspect_ratios"] = rec_aspect_ratios

    # ---- plots ----
    _plot_hist(sizes, "GT box size (sqrt(w*h), px)", out_dir / "gt_size_distribution.png",
               bins=60, vlines=pivots)
    _plot_hist(ratios, "GT aspect ratio (w/h)", out_dir / "gt_aspect_distribution.png",
               bins=50, log=False)
    _plot_hist(counts, "Lesions per image", out_dir / "gt_per_image.png", bins=40)

    # ---- save ----
    out_path = out_dir / "gt_size_stats.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {out_dir}/gt_size_distribution.png + 2 more")

    _print_report(summary)
    return summary


def _plot_hist(values, title, path, bins=50, log=False, vlines=()):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins, color="#3a7", edgecolor="black", alpha=0.85)
    for v in vlines:
        ax.axvline(v, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xlabel(title)
    ax.set_ylabel("count")
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _print_report(s: dict):
    print("\n=== Summary ===")
    print(f"  Images       : {s['n_images']:,}")
    print(f"  Boxes        : {s['n_boxes']:,}")
    print(f"  Median image : {s['image_size_median']}")
    print(f"\n  Box size (px, sqrt(w*h)):")
    for k, v in s["box_size_px"].items():
        print(f"    {k:>4}: {v:6.1f}")
    print(f"\n  Aspect ratio (w/h):")
    for k, v in s["aspect_ratio"].items():
        print(f"    {k:>4}: {v:6.2f}")
    print(f"\n  Boxes per image:")
    for k, v in s["boxes_per_image"].items():
        print(f"    {k:>4}: {v:6.1f}")
    print(f"    max : {s['boxes_per_image_max']}")
    print("\n=== Detector tuning recommendations ===")
    print(f"\nFaster R-CNN anchor_sizes (data-driven):")
    print(f"    {s['recommended_anchor_sizes']}")
    print(f"  Compare to current code:")
    print(f"    ((4, 6), (8, 12), (16, 24), (32, 48), (64, 96))")
    print(f"\nFaster R-CNN aspect_ratios (data-driven):")
    print(f"    {s['recommended_aspect_ratios']}")
    print(f"  Compare to current code:")
    print(f"    (0.5, 1.0, 2.0)")
    p95 = s["boxes_per_image"]["p95"]
    print(f"\nbox_detections_per_img: current=300, p95 boxes/img={p95:.0f}",
          "(OK)" if p95 < 250 else "(consider raising to 500)")
    p50_size = s["box_size_px"]["p50"]
    print(f"\nimgsz: median lesion size is {p50_size:.1f} px in the source image.",
          "Bumping training imgsz to 1280 effectively makes the median lesion",
          f"~{p50_size * 1280 / s['image_size_median'][0]:.0f} px to the model.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco", default="data/acne04/coco/train/_annotations.coco.json")
    p.add_argument("--out-dir", default="outputs/eda")
    args = p.parse_args()
    analyze(args.coco, args.out_dir)
