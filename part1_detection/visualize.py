"""
Visualize detector predictions on ACNE04 test images.

Produces side-by-side overlays:
    - green boxes  = ground truth
    - red boxes    = model predictions (with confidence)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO


def _draw_boxes(ax, boxes_xywh, color, scores=None, label=None):
    for i, (x, y, w, h) in enumerate(boxes_xywh):
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        if scores is not None:
            ax.text(
                x,
                max(y - 3, 0),
                f"{scores[i]:.2f}",
                color=color,
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.6, pad=0.5, edgecolor="none"),
            )
    if label:
        ax.set_title(label, fontsize=10)


def _yolo_predict_one(model, img_path, conf=0.25, imgsz=640):
    r = model.predict(img_path, conf=conf, imgsz=imgsz, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    xyxy = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    xywh = np.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
    return xywh, scores


def _frcnn_predict_one(model, img_path, conf=0.25, device=None):
    from torchvision.transforms import functional as F

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(img_path).convert("RGB")
    x = F.to_tensor(img).to(device)
    with torch.no_grad():
        out = model([x])[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    keep = scores >= conf
    boxes = boxes[keep]
    scores = scores[keep]
    xywh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1) if len(boxes) else np.zeros((0, 4))
    return xywh, scores


def visualize_predictions(
    models: dict[str, dict[str, Any]],
    coco_root: str | os.PathLike,
    split: str = "test",
    n_images: int = 6,
    out_dir: str | os.PathLike = "outputs/viz",
    seed: int = 0,
    conf: float = 0.25,
):
    """Render n_images x (1 + len(models)) grid: GT plus each model's predictions.

    Parameters
    ----------
    models : mapping from display name -> dict with keys:
                "kind" in {"yolo","rtdetr","frcnn"}
                "weights" : path
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_json = Path(coco_root) / split / "_annotations.coco.json"
    coco = COCO(str(gt_json))
    img_ids = coco.getImgIds()
    rng.shuffle(img_ids)
    img_ids = img_ids[:n_images]
    img_dir = Path(coco_root) / split

    # Lazy-load each model once
    loaded = {}
    for name, spec in models.items():
        kind = spec["kind"]
        if kind in ("yolo", "rtdetr"):
            from ultralytics import RTDETR, YOLO

            loaded[name] = (kind, (RTDETR if kind == "rtdetr" else YOLO)(str(spec["weights"])))
        elif kind == "frcnn":
            from .train_faster_rcnn import load_model

            loaded[name] = (kind, load_model(spec["weights"]))
        else:
            raise ValueError(f"unknown kind {kind}")

    n_cols = 1 + len(models)
    fig, axes = plt.subplots(n_images, n_cols, figsize=(4 * n_cols, 4 * n_images))
    if n_images == 1:
        axes = np.expand_dims(axes, 0)

    for r, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        img_path = img_dir / info["file_name"]
        img = np.array(Image.open(img_path).convert("RGB"))

        # GT
        gt_boxes = [a["bbox"] for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))]
        ax = axes[r, 0]
        ax.imshow(img)
        _draw_boxes(ax, gt_boxes, color="lime", label=f"Ground truth ({info['file_name']})")
        ax.axis("off")

        # Each model
        for c, (name, (kind, model)) in enumerate(loaded.items(), start=1):
            ax = axes[r, c]
            ax.imshow(img)
            if kind in ("yolo", "rtdetr"):
                xywh, scores = _yolo_predict_one(model, str(img_path), conf=conf)
            else:
                xywh, scores = _frcnn_predict_one(model, str(img_path), conf=conf)
            _draw_boxes(ax, xywh, color="red", scores=scores, label=name)
            ax.axis("off")

    plt.tight_layout()
    out_path = out_dir / f"predictions_{split}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)
    return out_path
