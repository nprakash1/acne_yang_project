"""
Evaluate the ACNE04-trained classifier on DermNet (cross-domain).

DermNet test set on Kaggle has ~20+ class folders. We binarize:
    - "acne"     <- folders matching ACNE_CLASS_KEYWORDS
    - "non-acne" <- everything else

We compute Accuracy, F1, AUROC, and a confusion matrix.

The DermNet test set is expected to live at:
    <root>/test/<class_folder>/*.jpg
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from .dataset import get_eval_transforms
from .domain_adapt import build_reference_mosaic, make_preprocessor
from .train_classifier import load_classifier, sample_label_blind_dermnet_paths


# Folder names in the Kaggle DermNet release that correspond to acne.
ACNE_CLASS_KEYWORDS = ("acne", "rosacea")


class DermNetBinaryDataset(Dataset):
    def __init__(self, root: str | os.PathLike, transform=None, preprocess=None):
        self.root = Path(root)
        self.transform = transform
        self.preprocess = preprocess
        self.samples: list[tuple[Path, int]] = []
        for cls_dir in sorted(self.root.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = 1 if any(k in cls_dir.name.lower() for k in ACNE_CLASS_KEYWORDS) else 0
            for p in cls_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.preprocess is not None:
            img = self.preprocess(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long), str(path)


def collate_with_paths(batch):
    xs, ys, ps = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(ps)


@torch.no_grad()
def predict(model, loader, device, tta: bool = False):
    model.eval()
    all_probs, all_y, all_paths = [], [], []
    for x, y, paths in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        if tta:
            x_flip = torch.flip(x, dims=[3])
            logits = (logits + model(x_flip)) / 2
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.numpy())
        all_paths.extend(paths)
    return np.concatenate(all_probs), np.concatenate(all_y), all_paths


def evaluate(
    weights: str,
    dermnet_root: str,
    test_subdir: str = "test",
    train_subdir: str = "train",
    n_unlabeled_for_da: int = 20,
    da_method: str = "reinhard",
    img_size: int = 224,
    batch_size: int = 64,
    tta: bool = True,
    out_dir: str = "outputs/dermnet_eval",
    device: str | None = None,
):
    """Run cross-domain evaluation. Returns metric dict, also saves preds.json."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build reference mosaic from <=20 unlabeled DermNet train images.
    # Sample LABEL-BLIND: uniform random over all train images regardless of
    # which folder they came from. (The previous implementation took one
    # image per class folder, which is accidentally label-stratified and
    # would technically constitute a label leak under a strict reading of
    # the assignment.)
    train_root = Path(dermnet_root) / train_subdir
    sample_paths = sample_label_blind_dermnet_paths(
        train_root, n=n_unlabeled_for_da
    )

    if sample_paths and da_method != "none":
        ref = build_reference_mosaic(sample_paths)
        preprocess = make_preprocessor(ref, method=da_method)
        print(f"[DA] using {da_method} with reference from {len(sample_paths)} DermNet train imgs")
    else:
        preprocess = None
        print("[DA] no preprocessing applied")

    eval_tf = get_eval_transforms(img_size=img_size)
    test_ds = DermNetBinaryDataset(
        Path(dermnet_root) / test_subdir, transform=eval_tf, preprocess=preprocess
    )
    print(f"DermNet test images: {len(test_ds)}")

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_with_paths,
    )

    model = load_classifier(weights, device=device)
    probs, y, paths = predict(model, loader, device, tta=tta)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "auroc": float(roc_auc_score(y, probs)) if len(set(y)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "da_method": da_method,
        "tta": tta,
    }
    print(json.dumps(metrics, indent=2))

    # Save per-image preds for downstream gradcam / inspection
    rows = [
        {"path": p, "label": int(yi), "prob_acne": float(pr), "pred": int(pd)}
        for p, yi, pr, pd in zip(paths, y, probs, preds)
    ]
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(rows, f)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--dermnet-root", required=True)
    p.add_argument("--da-method", default="reinhard", choices=["none", "histogram", "reinhard"])
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--out", default="outputs/dermnet_eval")
    args = p.parse_args()
    evaluate(
        weights=args.weights,
        dermnet_root=args.dermnet_root,
        da_method=args.da_method,
        tta=not args.no_tta,
        out_dir=args.out,
    )
