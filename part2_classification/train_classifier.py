"""
Train a binary acne / non-acne classifier on ACNE04 patches.

Selects best checkpoint by validation AUROC.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .dataset import (
    PatchFolder,
    class_weights,
    get_eval_transforms,
    get_train_transforms,
)
from .model import build_face_resnet, build_resnet50


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict[str, float]:
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y).numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == y).mean())
    try:
        auroc = float(roc_auc_score(y, probs))
    except ValueError:
        auroc = float("nan")
    return {"acc": acc, "auroc": auroc}


def train(
    patches_root: str | os.PathLike,
    out_dir: str | os.PathLike = "outputs/classifier",
    backbone: str = "resnet50",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    img_size: int = 224,
    heavy_da: bool = True,
    device: str | None = None,
    num_workers: int = 4,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = get_train_transforms(img_size=img_size, heavy_da=heavy_da)
    eval_tf = get_eval_transforms(img_size=img_size)

    train_ds = PatchFolder(Path(patches_root) / "train", transform=train_tf)
    val_ds = PatchFolder(Path(patches_root) / "val", transform=eval_tf)
    print(f"train patches: {len(train_ds)}  |  val patches: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if backbone == "resnet50":
        model = build_resnet50(num_classes=2, pretrained=True)
    elif backbone == "vggface2":
        model = build_face_resnet(num_classes=2)
    else:
        raise ValueError(backbone)
    model.to(device)

    weights = class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auroc = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()
        train_loss = running / max(n, 1)

        val_metrics = evaluate_loader(model, val_loader, device)
        print(
            f"epoch {epoch:02d}  loss {train_loss:.4f}  "
            f"val_acc {val_metrics['acc']:.4f}  val_auroc {val_metrics['auroc']:.4f}"
        )
        history.append({"epoch": epoch, "loss": train_loss, **val_metrics})

        torch.save({"model": model.state_dict(), "backbone": backbone, "epoch": epoch}, out_dir / "last.pt")
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(
                {"model": model.state_dict(), "backbone": backbone, "epoch": epoch},
                out_dir / "best.pt",
            )
            print(f"  -> new best val AUROC {best_auroc:.4f}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[classifier] best AUROC = {best_auroc:.4f}, weights at {out_dir/'best.pt'}")
    return out_dir / "best.pt"


def load_classifier(weights: str | os.PathLike, device: str | None = None) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights, map_location=device)
    backbone = ckpt.get("backbone", "resnet50")
    model = build_resnet50(num_classes=2, pretrained=False) if backbone == "resnet50" else build_face_resnet(num_classes=2)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patches", required=True)
    p.add_argument("--out", default="outputs/classifier")
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "vggface2"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--no-heavy-da", action="store_true")
    args = p.parse_args()
    train(
        patches_root=args.patches,
        out_dir=args.out,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        heavy_da=not args.no_heavy_da,
    )
