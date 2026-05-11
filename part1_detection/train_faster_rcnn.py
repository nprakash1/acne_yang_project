"""
Train Faster R-CNN with ResNet-50-FPN on ACNE04.

Classic two-stage detector via torchvision. We keep COCO-pretrained
weights and replace only the final box predictor head for our 1 class.
Anchor sizes are *shrunk* relative to the COCO defaults because acne
lesions are typically 5-30 px across.

Usage:
    from part1_detection.train_faster_rcnn import train
    train(coco_root="data/acne04/coco", epochs=25)
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

from .data import AcneCocoDataset, collate_fn


# ----------------------------- transforms ----------------------------- #


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1).item() < self.p:
            image = F.hflip(image)
            w = image.shape[-1]
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_train_transforms():
    return Compose([ToTensor(), RandomHorizontalFlip(0.5)])


def get_val_transforms():
    return Compose([ToTensor()])


# ----------------------------- model ---------------------------------- #


def build_model(num_classes: int = 2, small_anchors: bool = True) -> FasterRCNN:
    """Build a Faster R-CNN model with ResNet-50-FPN.

    num_classes includes the background class, so for ACNE04 (1 fg class) it's 2.
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    if small_anchors:
        # Default torchvision anchor sizes are (32, 64, 128, 256, 512) -- way too
        # large for acne lesions (~5-30 px). Shrink them while keeping FPN's
        # one-size-per-level convention.
        anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios
        )
        # The RPN head depends on the number of anchors per location -- rebuild it.
        in_channels = model.backbone.out_channels
        num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
        from torchvision.models.detection.rpn import RPNHead

        model.rpn.head = RPNHead(in_channels, num_anchors)

    # Replace the final classifier head for our number of classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ----------------------------- training ------------------------------- #


def train_one_epoch(model, optimizer, loader, device, epoch, print_every=20):
    model.train()
    running = 0.0
    n = 0
    t0 = time.time()
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if not math.isfinite(loss.item()):
            print(f"WARN: non-finite loss {loss.item()} -- skipping batch")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running += loss.item() * len(images)
        n += len(images)
        if i % print_every == 0:
            print(
                f"  epoch {epoch} step {i}/{len(loader)} "
                f"loss {loss.item():.4f} ({running/max(n,1):.4f} avg)"
            )
    print(f"  epoch {epoch} done in {time.time()-t0:.1f}s, avg loss {running/max(n,1):.4f}")
    return running / max(n, 1)


def train(
    coco_root: str | os.PathLike,
    epochs: int = 25,
    batch_size: int = 4,
    lr: float = 5e-3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    out_dir: str | os.PathLike = "outputs/faster_rcnn",
    device: str | None = None,
    num_workers: int = 2,
    train_split: str = "train",
    val_split: str = "valid",
):
    """Train Faster R-CNN on ACNE04 in COCO format."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AcneCocoDataset(coco_root, split=train_split, transforms=get_train_transforms())
    val_ds = AcneCocoDataset(coco_root, split=val_split, transforms=get_val_transforms())
    print(f"train images: {len(train_ds)}  |  val images: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=2, small_anchors=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        # Save latest + best
        torch.save(
            {"model": model.state_dict(), "epoch": epoch},
            out_dir / "last.pth",
        )
        if loss < best_loss:
            best_loss = loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch},
                out_dir / "best.pth",
            )
            print(f"  -> new best train loss {best_loss:.4f}, saved best.pth")

    print(f"[Faster R-CNN] best weights: {out_dir/'best.pth'}")
    return model


def load_model(weights: str | os.PathLike, device: str | None = None) -> FasterRCNN:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2, small_anchors=True)
    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--out", default="outputs/faster_rcnn")
    args = p.parse_args()
    train(
        coco_root=args.coco_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out,
    )
