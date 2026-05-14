"""Train Ultralytics detectors on ACNE04: YOLOv8 and RT-DETR.

One small wrapper covers both modern one-stage CNN (YOLOv8) and transformer
(DETR/DINO-family) detectors. Per-model defaults live in ``ULTRALYTICS_MODELS``
so the comparison stays explicit and easy to tweak.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal


# Larger imgsz (1024) is the single biggest lever for acne detection because
# lesions are only 5-30 px wide in the source 640px images. We pair that with
# longer training and slightly smaller batches to keep VRAM in budget.
ULTRALYTICS_MODELS: dict[str, dict] = {
    "yolo": {
        # yolov8m: 26M params vs yolov8s's 11M. ~+5 mAP on COCO and typically
        # a larger gain on dense small-object datasets like ACNE04 because
        # the deeper backbone preserves more detail at the P3/P4 levels.
        "weights": "yolov8m.pt",
        "project": "outputs/yolov8",
        "epochs": 150,
        # Batch dropped 12 -> 8 to keep VRAM under 14 GB at imgsz=1024 on a T4.
        "batch": 8,
        "imgsz": 1024,
        # YOLO-friendly: mosaic + mixup + HSV jitter
        "aug": dict(mosaic=1.0, mixup=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4),
    },
    "rtdetr": {
        "weights": "rtdetr-l.pt",
        "project": "outputs/rtdetr",
        "epochs": 120,
        "batch": 6,
        "imgsz": 1024,
        # DETR-family detectors are usually more stable with milder aug
        "aug": dict(mosaic=0.0, mixup=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3),
    },
}



def train_ultralytics(
    kind: Literal["yolo", "rtdetr"],
    data: str,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    model_name: str | None = None,
    project: str | None = None,
    name: str = "acne",
    device: str | int | None = None,
    patience: int = 30,
):
    """Train YOLOv8 or RT-DETR. Returns the Ultralytics results object."""
    if kind not in ULTRALYTICS_MODELS:
        raise ValueError(f"kind must be one of {list(ULTRALYTICS_MODELS)}")

    cfg = ULTRALYTICS_MODELS[kind]
    model_name = model_name or cfg["weights"]
    project = project or cfg["project"]
    epochs = epochs or cfg["epochs"]
    batch = batch or cfg["batch"]
    imgsz = imgsz or cfg.get("imgsz", 640)

    if kind == "rtdetr":
        from ultralytics import RTDETR as Model
    else:
        from ultralytics import YOLO as Model


    model = Model(model_name)
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
        device=device,
        patience=patience,
        fliplr=0.5,
        flipud=0.0,
        **cfg["aug"],
    )
    print(f"[{kind}] best weights: {Path(project) / name / 'weights' / 'best.pt'}")
    return results


# Convenience aliases keep notebook code reading naturally.
def train_yolo(data: str, **kw):
    return train_ultralytics("yolo", data=data, **kw)


def train_rtdetr(data: str, **kw):
    return train_ultralytics("rtdetr", data=data, **kw)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["yolo", "rtdetr"])
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    train_ultralytics(
        args.kind,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
        device=args.device,
    )
