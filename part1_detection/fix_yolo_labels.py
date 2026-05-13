"""Remap multi-class YOLO labels to single-class (id 0).

The Roboflow ACNE04 YOLO export uses class ids 0/1/2/3 for the four severity
levels (levle0..levle3), but our pipeline treats acne as a single class.
Ultralytics silently drops every image whose label class id exceeds nc-1, so
~75% of training data was being thrown away.

This script rewrites every YOLO label .txt under the given root, replacing
the leading class id with 0. Safe to re-run (idempotent).

Usage:
    python -m part1_detection.fix_yolo_labels --root data/acne04/yolov8
"""

from __future__ import annotations
import argparse
from pathlib import Path


def remap_labels_to_single_class(root: str) -> dict:
    root = Path(root)
    label_files = list(root.rglob("*.txt"))
    label_files = [p for p in label_files if "labels" in p.parts or "labels" in str(p)]

    # Roboflow YOLO exports put labels in `<split>/labels/*.txt`
    if not label_files:
        # Fallback: any .txt under root
        label_files = list(root.rglob("*.txt"))

    counts = {"files_total": len(label_files), "files_changed": 0,
              "lines_changed": 0, "labels_cache_removed": 0}

    for txt in label_files:
        # Skip Ultralytics cache files
        if txt.name == "labels.cache":
            continue
        try:
            lines = txt.read_text().splitlines()
        except Exception:
            continue
        new_lines = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                new_lines.append(line)
                continue
            if parts[0] != "0":
                parts[0] = "0"
                counts["lines_changed"] += 1
                changed = True
            new_lines.append(" ".join(parts))
        if changed:
            txt.write_text("\n".join(new_lines) + "\n")
            counts["files_changed"] += 1

    # Nuke Ultralytics label caches so the remap actually takes effect on
    # the next training run -- otherwise Ultralytics will reuse the cached
    # "corrupt" verdict.
    for cache in root.rglob("labels.cache"):
        cache.unlink()
        counts["labels_cache_removed"] += 1

    print(f"[fix_yolo_labels] root           : {root}")
    print(f"[fix_yolo_labels] label files    : {counts['files_total']}")
    print(f"[fix_yolo_labels] files changed  : {counts['files_changed']}")
    print(f"[fix_yolo_labels] lines remapped : {counts['lines_changed']}")
    print(f"[fix_yolo_labels] caches removed : {counts['labels_cache_removed']}")
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/acne04/yolov8",
                   help="Root of the Roboflow YOLO export (contains train/valid/test).")
    args = p.parse_args()
    remap_labels_to_single_class(args.root)
