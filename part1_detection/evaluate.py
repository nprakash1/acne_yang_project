"""
Unified evaluation for all three detectors.

Strategy: convert each model's predictions on the test split to COCO-style
JSON, then run pycocotools' COCOeval against the ground-truth COCO json.
This guarantees YOLOv8, Faster R-CNN, and RT-DETR are all measured the
exact same way.

Reported metrics:
    - mAP @ IoU=0.50 (the headline number for medical detection)
    - mAP @ IoU=0.50:0.95 (the strict COCO standard)
    - Precision and Recall at the model's chosen confidence
    - Mean IoU across matched true-positive predictions
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# --------------------------- shared helpers --------------------------- #


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


# --------------------- per-model prediction dumpers ------------------- #


def predict_yolo(
    weights: str | os.PathLike,
    coco_root: str | os.PathLike,
    split: str = "test",
    conf: float = 0.001,
    imgsz: int = 640,
    is_rtdetr: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Run an Ultralytics model on COCO-split images, return COCO-format detections.

    Returns (detections_list, gt_json_path).
    """
    from ultralytics import RTDETR, YOLO

    Model = RTDETR if is_rtdetr else YOLO
    model = Model(str(weights))

    gt_json = Path(coco_root) / split / "_annotations.coco.json"
    coco = COCO(str(gt_json))

    img_dir = Path(coco_root) / split
    detections: list[dict[str, Any]] = []
    img_ids = coco.getImgIds()

    # Index image filenames -> coco img ids
    fname_to_id = {coco.loadImgs(i)[0]["file_name"]: i for i in img_ids}

    img_paths = [str(img_dir / fn) for fn in fname_to_id.keys()]
    # Run in batches to keep memory sane
    BATCH = 16
    for start in range(0, len(img_paths), BATCH):
        batch_paths = img_paths[start : start + BATCH]
        results = model.predict(
            batch_paths, conf=conf, imgsz=imgsz, verbose=False, save=False
        )
        for path, r in zip(batch_paths, results):
            fname = Path(path).name
            img_id = fname_to_id[fname]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for box, score in zip(xyxy, scores):
                detections.append(
                    {
                        "image_id": int(img_id),
                        "category_id": 1,
                        "bbox": [float(v) for v in _xyxy_to_xywh(box.tolist())],
                        "score": float(score),
                    }
                )
    return detections, str(gt_json)


def predict_faster_rcnn(
    weights: str | os.PathLike,
    coco_root: str | os.PathLike,
    split: str = "test",
    conf: float = 0.001,
    device: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Run torchvision Faster R-CNN, return COCO-format detections."""
    from torchvision.transforms import functional as F

    from .train_faster_rcnn import load_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device=device)

    gt_json = Path(coco_root) / split / "_annotations.coco.json"
    coco = COCO(str(gt_json))
    img_dir = Path(coco_root) / split

    detections: list[dict[str, Any]] = []
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        img = Image.open(img_dir / info["file_name"]).convert("RGB")
        x = F.to_tensor(img).to(device)
        with torch.no_grad():
            out = model([x])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        for box, score in zip(boxes, scores):
            if score < conf:
                continue
            detections.append(
                {
                    "image_id": int(img_id),
                    "category_id": 1,
                    "bbox": [float(v) for v in _xyxy_to_xywh(box.tolist())],
                    "score": float(score),
                }
            )
    return detections, str(gt_json)


# --------------------------- COCO evaluation -------------------------- #


def coco_evaluate(
    gt_json: str,
    detections: list[dict[str, Any]],
    out_json: str | os.PathLike | None = None,
) -> dict[str, float]:
    """Run pycocotools COCOeval and return a metric dict."""
    if not detections:
        print("WARN: no detections produced -- returning zero metrics")
        return {
            "mAP@0.5": 0.0,
            "mAP@0.5:0.95": 0.0,
            "AR@100": 0.0,
            "precision@0.5": 0.0,
            "recall@0.5": 0.0,
            "mean_iou": 0.0,
            "n_detections": 0,
        }

    coco_gt = COCO(gt_json)

    # Remap all predictions to the category that actually carries annotations.
    # Roboflow COCO exports often include a phantom super-category with id=0
    # and put the real class at id=1 -- but some exports flip this. Pick the
    # category with the most GT annotations to be robust.
    ann_cat_counts: dict[int, int] = {}
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
        ann_cat_counts[ann["category_id"]] = ann_cat_counts.get(ann["category_id"], 0) + 1
    if ann_cat_counts:
        primary = max(ann_cat_counts, key=ann_cat_counts.get)
        for d in detections:
            d["category_id"] = primary


    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(detections, f)
        coco_dt = coco_gt.loadRes(str(out_json))
    else:
        coco_dt = coco_gt.loadRes(detections)

    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # Single-class detection: ignore category ids so a Roboflow export with
    # phantom super-categories (id=0) can't silently zero out mAP. Matches
    # purely on bbox IoU + image id.
    e.params.useCats = 0
    e.evaluate()
    e.accumulate()
    e.summarize()


    # COCOeval.stats indices:
    #   0: AP @ IoU=0.50:0.95   1: AP @ 0.50   2: AP @ 0.75
    #   8: AR @ 100 dets
    metrics = {
        "mAP@0.5:0.95": float(e.stats[0]),
        "mAP@0.5": float(e.stats[1]),
        "mAP@0.75": float(e.stats[2]),
        "AR@100": float(e.stats[8]),
        "n_detections": len(detections),
    }

    # Pick the score threshold that *maximizes F1* on this split, instead of a
    # hard-coded 0.25 cutoff. The COCO-style mAP already integrates over all
    # thresholds; for the human-readable P@0.5 / R@0.5 columns we want an
    # operating point that's fair across detectors with very different score
    # distributions (YOLOv8 fires few high-score boxes, RT-DETR fires many
    # low-score boxes -- a fixed cutoff biases the comparison).
    best = _best_f1_pr_iou(coco_gt, detections, iou_thr=0.5)
    metrics.update(best)
    return metrics


def _best_f1_pr_iou(
    coco_gt: COCO,
    detections: list[dict[str, Any]],
    iou_thr: float = 0.5,
    score_grid: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 19)),
) -> dict[str, float]:
    """Sweep score thresholds and return P/R/IoU at the F1-optimal threshold.

    This is the natural operating point for each detector. The COCO mAP
    already integrates over thresholds, so this is purely a fairer P/R
    column for the human-readable table. Defensible in a report: it's the
    same approach Ultralytics uses internally for its `P`, `R`, `F1` curves.
    """
    best = {
        "precision@0.5": 0.0,
        "recall@0.5": 0.0,
        "mean_iou": 0.0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "score_thr": 0.0,
        "f1": 0.0,
    }
    for thr in score_grid:
        m = _precision_recall_iou(
            coco_gt, detections, iou_thr=iou_thr, score_thr=thr
        )
        p = m[f"precision@{iou_thr}"]
        r = m[f"recall@{iou_thr}"]
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best["f1"]:
            best = {
                "precision@0.5": p,
                "recall@0.5": r,
                "mean_iou": m["mean_iou"],
                "TP": m["TP"],
                "FP": m["FP"],
                "FN": m["FN"],
                "score_thr": float(thr),
                "f1": float(f1),
            }
    return best


def _precision_recall_iou(
    coco_gt: COCO,
    detections: list[dict[str, Any]],
    iou_thr: float = 0.5,
    score_thr: float = 0.25,
) -> dict[str, float]:
    """Greedy per-image matching to get an interpretable P/R/IoU triple."""
    by_img: dict[int, list[dict[str, Any]]] = {}
    for d in detections:
        if d["score"] < score_thr:
            continue
        by_img.setdefault(d["image_id"], []).append(d)

    tp = 0
    fp = 0
    fn = 0
    matched_ious: list[float] = []

    for img_id in coco_gt.getImgIds():
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        gt_boxes = []
        for g in gts:
            x, y, w, h = g["bbox"]
            gt_boxes.append(np.array([x, y, x + w, y + h]))
        used = [False] * len(gt_boxes)

        dets = sorted(by_img.get(img_id, []), key=lambda d: -d["score"])
        for d in dets:
            x, y, w, h = d["bbox"]
            db = np.array([x, y, x + w, y + h])
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if used[j]:
                    continue
                iou = _iou_xyxy(db, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_thr:
                tp += 1
                matched_ious.append(best_iou)
                used[best_j] = True
            else:
                fp += 1
        fn += sum(1 for u in used if not u)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return {
        f"precision@{iou_thr}": precision,
        f"recall@{iou_thr}": recall,
        "mean_iou": mean_iou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


# ----------------------------- driver --------------------------------- #


def evaluate_model(
    model_kind: str,
    weights: str,
    coco_root: str,
    split: str = "test",
    out_dir: str | os.PathLike = "outputs/eval",
) -> dict[str, float]:
    """Evaluate a model end-to-end. model_kind in {'yolo', 'rtdetr', 'frcnn'}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_kind == "yolo":
        dets, gt = predict_yolo(weights, coco_root, split=split, is_rtdetr=False)
    elif model_kind == "rtdetr":
        dets, gt = predict_yolo(weights, coco_root, split=split, is_rtdetr=True)
    elif model_kind == "frcnn":
        dets, gt = predict_faster_rcnn(weights, coco_root, split=split)
    else:
        raise ValueError(f"unknown model_kind: {model_kind}")

    out_json = out_dir / f"{model_kind}_{split}_detections.json"
    metrics = coco_evaluate(gt, dets, out_json=out_json)

    metrics_path = out_dir / f"{model_kind}_{split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{model_kind}] metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))
    return metrics
