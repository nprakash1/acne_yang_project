"""
Visualize where the classifier looks on DermNet using Grad-CAM.

Saves a grid of N DermNet images with Grad-CAM heatmaps overlaid,
labelled with (true class, predicted prob).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .dataset import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from .domain_adapt import build_reference_mosaic, make_preprocessor
from .model import get_target_layers_for_gradcam
from .train_classifier import load_classifier


def _denormalize_for_display(img_tensor: torch.Tensor) -> np.ndarray:
    arr = img_tensor.cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(arr, 0, 1)


def visualize_gradcam(
    weights: str,
    predictions_json: str,
    out_dir: str = "outputs/gradcam",
    n_samples: int = 10,
    img_size: int = 224,
    da_method: str = "reinhard",
    dermnet_train_root: str | None = None,
    seed: int = 0,
    device: str | None = None,
    sampling: str = "stratified",  # "stratified" -> mix of TP/FP/TN/FN
):
    """Render Grad-CAM heatmaps for n_samples DermNet predictions.

    Reads predictions.json produced by evaluate_dermnet.py and picks a
    diverse set (TP/FP/TN/FN) by default.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    with open(predictions_json) as f:
        rows = json.load(f)

    # Stratified pick: 2-3 from each (TP, FP, TN, FN) for visual diversity
    bins: dict[str, list[dict]] = {"TP": [], "FP": [], "TN": [], "FN": []}
    for r in rows:
        if r["label"] == 1 and r["pred"] == 1:
            bins["TP"].append(r)
        elif r["label"] == 0 and r["pred"] == 1:
            bins["FP"].append(r)
        elif r["label"] == 0 and r["pred"] == 0:
            bins["TN"].append(r)
        else:
            bins["FN"].append(r)

    selected: list[dict] = []
    if sampling == "stratified":
        per_bin = max(1, n_samples // 4)
        for k in ("TP", "FP", "TN", "FN"):
            rng.shuffle(bins[k])
            selected.extend(bins[k][:per_bin])
        # Pad to n_samples from any non-empty bin
        leftover = [r for k in bins for r in bins[k] if r not in selected]
        rng.shuffle(leftover)
        selected.extend(leftover[: max(0, n_samples - len(selected))])
        selected = selected[:n_samples]
    else:
        selected = rng.sample(rows, min(n_samples, len(rows)))

    # Build the same DA preprocessor we used at eval, so heatmaps reflect
    # what the model actually saw.
    preprocess = None
    if dermnet_train_root and da_method != "none":
        train_root = Path(dermnet_train_root)
        sample_paths = []
        for cls_dir in sorted(train_root.iterdir()):
            if cls_dir.is_dir():
                for p in cls_dir.iterdir():
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        sample_paths.append(p)
                        break
            if len(sample_paths) >= 20:
                break
        if sample_paths:
            ref = build_reference_mosaic(sample_paths[:20])
            preprocess = make_preprocessor(ref, method=da_method)

    eval_tf = get_eval_transforms(img_size=img_size)
    model = load_classifier(weights, device=device)
    target_layers = get_target_layers_for_gradcam(model)
    cam = GradCAM(model=model, target_layers=target_layers)

    n = len(selected)
    cols = 5
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 3.2 * rows_n))
    axes = np.atleast_2d(axes)

    for i, row in enumerate(selected):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        img = np.array(Image.open(row["path"]).convert("RGB"))
        if preprocess is not None:
            img = preprocess(img)
        x = eval_tf(image=img)["image"].unsqueeze(0).to(device)
        rgb_for_show = _denormalize_for_display(x[0])

        target_class = int(row["pred"])
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])[0]
        overlay = show_cam_on_image(rgb_for_show, grayscale_cam, use_rgb=True)

        ax.imshow(overlay)
        kind = (
            "TP" if row["label"] == 1 and row["pred"] == 1
            else "FP" if row["label"] == 0 and row["pred"] == 1
            else "TN" if row["label"] == 0 and row["pred"] == 0
            else "FN"
        )
        ax.set_title(
            f"{kind}  p(acne)={row['prob_acne']:.2f}\n{Path(row['path']).parent.name}",
            fontsize=8,
        )
        ax.axis("off")

    # blank any unused axes
    for j in range(n, rows_n * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    out_path = out_dir / "gradcam_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--predictions", required=True, help="predictions.json from evaluate_dermnet")
    p.add_argument("--out", default="outputs/gradcam")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--dermnet-train", default=None)
    p.add_argument("--da-method", default="reinhard")
    args = p.parse_args()
    visualize_gradcam(
        weights=args.weights,
        predictions_json=args.predictions,
        out_dir=args.out,
        n_samples=args.n,
        dermnet_train_root=args.dermnet_train,
        da_method=args.da_method,
    )
