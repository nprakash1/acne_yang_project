"""
Train a binary acne / non-acne classifier on ACNE04 patches.

Selects best checkpoint by validation AUROC.

Supports optional source-side DermNet normalization during training: a
fraction of ACNE04 crops are Reinhard- / histogram-matched to a 20-image
DermNet reference mosaic so the source pixel distribution overlaps the
target one at training time. The 20-image budget is sampled
*label-blind* (uniformly over all DermNet train images regardless of folder)
to comply with the brief.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
    make_stochastic_preprocess,
)
from .domain_adapt import build_reference_mosaic, make_preprocessor
from .model import build_face_resnet, build_resnet50


def sample_label_blind_dermnet_paths(
    dermnet_train_root: str | os.PathLike,
    n: int = 20,
    seed: int = 0,
) -> list[Path]:
    """Sample `n` DermNet train images uniformly at random across all classes.

    This is the brief-compliant way to spend the 20-image target-domain
    budget: we never read folder names as labels, we just collect every
    image into one flat pool and sample uniformly.
    """
    root = Path(dermnet_train_root)
    if not root.exists():
        return []
    all_paths: list[Path] = []
    for d in root.rglob("*"):
        if d.is_file() and d.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            all_paths.append(d)
    if not all_paths:
        return []
    rng = random.Random(seed)
    rng.shuffle(all_paths)
    return all_paths[:n]


def build_source_side_preprocessor(
    dermnet_train_root: str | os.PathLike | None,
    method: str = "reinhard",
    n_ref: int = 20,
    apply_p: float = 0.5,
    seed: int = 0,
):
    """Return a stochastic uint8-RGB->uint8-RGB preprocessor or None.

    If method=='none' or no DermNet root is provided, returns None.
    Otherwise applies Reinhard/histogram match (with prob apply_p per call)
    using a label-blind 20-image DermNet reference mosaic.
    """
    if method == "none" or not dermnet_train_root:
        return None
    sample_paths = sample_label_blind_dermnet_paths(
        dermnet_train_root, n=n_ref, seed=seed
    )
    if not sample_paths:
        print(
            f"[source-DA] no DermNet train images found under {dermnet_train_root}; "
            "skipping source-side normalization."
        )
        return None
    ref = build_reference_mosaic(sample_paths)
    base = make_preprocessor(ref, method=method)
    print(
        f"[source-DA] applying {method} with prob={apply_p:.2f} during training, "
        f"reference built from {len(sample_paths)} label-blind DermNet train imgs"
    )
    return make_stochastic_preprocess(base, p=apply_p)


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
    color_shortcut_break: bool = True,
    source_da_method: str = "none",  # "none" | "reinhard" | "histogram"
    dermnet_train_root: str | os.PathLike | None = None,
    source_da_p: float = 0.5,
    device: str | None = None,
    num_workers: int = 4,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = get_train_transforms(
        img_size=img_size,
        heavy_da=heavy_da,
        color_shortcut_break=color_shortcut_break,
    )
    eval_tf = get_eval_transforms(img_size=img_size)

    # Source-side DA: optionally Reinhard-normalize some ACNE04 training crops
    # toward DermNet color statistics. Val set is left untouched so val AUROC
    # tracks in-domain performance.
    train_preprocess = build_source_side_preprocessor(
        dermnet_train_root=dermnet_train_root,
        method=source_da_method,
        apply_p=source_da_p,
    )

    train_ds = PatchFolder(
        Path(patches_root) / "train",
        transform=train_tf,
        preprocess=train_preprocess,
    )
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
    p.add_argument(
        "--no-color-break",
        action="store_true",
        help="Disable ToGray/ChannelShuffle/RGBShift augmentations (color-shortcut breakers).",
    )
    p.add_argument(
        "--source-da-method",
        default="none",
        choices=["none", "reinhard", "histogram"],
        help="Apply DermNet-style color normalization to a fraction of training images.",
    )
    p.add_argument(
        "--dermnet-train",
        default=None,
        help="Path to DermNet train root. Required if --source-da-method != none.",
    )
    p.add_argument("--source-da-p", type=float, default=0.5)
    args = p.parse_args()
    train(
        patches_root=args.patches,
        out_dir=args.out,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        heavy_da=not args.no_heavy_da,
        color_shortcut_break=not args.no_color_break,
        source_da_method=args.source_da_method,
        dermnet_train_root=args.dermnet_train,
        source_da_p=args.source_da_p,
    )
