"""
ACNE04 dataset utilities.

The ACNE04 dataset is hosted on Roboflow Universe:
    https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/

This module provides:
    - download_acne04(...): pulls the dataset in YOLOv8 format AND COCO format
    - build_yaml(...): writes the data YAML used by Ultralytics
    - AcneCocoDataset: torchvision-style Dataset for Faster R-CNN training
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


def download_acne04(
    out_dir: str | os.PathLike = "data/acne04",
    api_key: str | None = None,
    fmt: str = "yolov8",
) -> Path:
    """Download ACNE04 from Roboflow Universe.

    Parameters
    ----------
    out_dir : where to extract the dataset
    api_key : Roboflow API key. Falls back to ROBOFLOW_API_KEY env var.
    fmt     : "yolov8" for Ultralytics models, "coco" for Faster R-CNN.

    Returns
    -------
    Path to the dataset root.
    """
    from roboflow import Roboflow

    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if api_key is None:
        raise RuntimeError(
            "Set ROBOFLOW_API_KEY env var or pass api_key=. "
            "Get one (free) at https://app.roboflow.com/settings/api"
        )

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("acne-vulgaris-detection").project("acne04-detection")
    version = project.version(1)
    dataset = version.download(fmt, location=str(out_dir / fmt))
    return Path(dataset.location)


def build_yaml(yolo_root: str | os.PathLike, yaml_path: str | os.PathLike) -> Path:
    """Write a YOLOv8 / RT-DETR data YAML pointing at the Roboflow export."""
    yolo_root = Path(yolo_root).resolve()
    yaml_path = Path(yaml_path).resolve()
    content = (
        f"path: {yolo_root}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: 1\n"
        f"names: ['acne']\n"
    )
    yaml_path.write_text(content)
    return yaml_path


class AcneCocoDataset(Dataset):
    """COCO-format ACNE04 dataset for torchvision Faster R-CNN.

    Expects the Roboflow COCO export, which produces a directory layout:
        <root>/train/_annotations.coco.json
        <root>/train/<image>.jpg
    """

    def __init__(self, root: str | os.PathLike, split: str = "train", transforms=None):
        self.root = Path(root) / split
        self.transforms = transforms

        ann_file = self.root / "_annotations.coco.json"
        with open(ann_file) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        # group annotations by image id
        self.anns_by_img: dict[int, list[dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # Roboflow exports often include a placeholder category id 0 -- drop it.
        self.cat_ids = sorted(
            c["id"] for c in coco["categories"] if c["name"].lower() != "acne-detection"
        )
        # Map COCO cat_id -> 1 (single class, since 0 is background in torchvision)
        self.cat_id_to_label = {cid: 1 for cid in self.cat_ids}

        self.image_ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_info = self.images[self.image_ids[idx]]
        img_path = self.root / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        anns = self.anns_by_img.get(img_info["id"], [])
        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])  # xyxy
            labels.append(self.cat_id_to_label.get(a["category_id"], 1))
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info["id"]]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def collate_fn(batch):
    """torchvision detection collate (variable-size targets)."""
    return tuple(zip(*batch))
