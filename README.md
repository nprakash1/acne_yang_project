# Acne Detection & Cross-Domain Classification (Yang Lab interview project)

End-to-end implementation of:

- **Part 1** — Acne lesion **detection** on ACNE04 with three detectors (YOLOv8, Faster R-CNN with ResNet50-FPN, RT-DETR).
- **Part 2** — Cross-domain **classification** (ACNE04 → DermNet) with progressive domain adaptation and Grad-CAM visualizations.

## Repo layout

```
acne_yang_project/
├── README.md
├── report.md                    # 1-2 page writeup
├── requirements.txt
├── part1_detection/
│   ├── data.py                  # Roboflow download + COCO Dataset
│   ├── train_ultralytics.py     # YOLOv8 (one-stage CNN) + RT-DETR (transformer)
│   ├── train_faster_rcnn.py     # two-stage CNN with shrunk anchors
│   ├── evaluate.py              # unified pycocotools mAP / P / R / IoU
│   └── visualize.py             # bbox overlays
├── part2_classification/
│   ├── make_patches.py          # build pos/neg patches from ACNE04
│   ├── dataset.py               # PatchFolder + albumentations transforms
│   ├── model.py                 # ResNet50 / VGGFace2 backbones
│   ├── train_classifier.py
│   ├── domain_adapt.py          # histogram match / Reinhard normalization
│   ├── evaluate_dermnet.py      # binary acne/non-acne eval on DermNet
│   └── gradcam.py               # 10-image stratified Grad-CAM grid
├── notebooks/
│   ├── 01_part1_detection.ipynb
│   └── 02_part2_classification.ipynb
└── outputs/                     # weights, metrics, viz (gitignored)
```

## Quick start (Colab — recommended)

1. Open `notebooks/01_part1_detection.ipynb` in Google Colab. Set runtime → GPU (T4 or better).
2. Get a free Roboflow API key from https://app.roboflow.com/settings/api and paste it in the first download cell.
3. Run all cells top-to-bottom. ~3 hours on a T4.
4. Repeat for `notebooks/02_part2_classification.ipynb`. You'll be prompted to upload your `kaggle.json` to download DermNet.

## Quick start (local)

Requires CUDA + Python 3.10+.

```bash
git clone <this-repo>
cd acne_yang_project
pip install -r requirements.txt

# Part 1
export ROBOFLOW_API_KEY=...
python -c "from part1_detection.data import download_acne04, build_yaml; \
    yolo=download_acne04(fmt='yolov8'); coco=download_acne04(fmt='coco'); \
    build_yaml(yolo, 'data/acne04.yaml')"

python -m part1_detection.train_ultralytics  yolo   --data data/acne04.yaml --epochs 100
python -m part1_detection.train_faster_rcnn         --coco-root data/acne04/coco --epochs 25
python -m part1_detection.train_ultralytics  rtdetr --data data/acne04.yaml --epochs 80

# Part 2
python -m part2_classification.make_patches  --coco-root data/acne04/coco --out data/acne04_patches
kaggle datasets download -d shubhamgoel27/dermnet -p data/dermnet --unzip
python -m part2_classification.train_classifier   --patches data/acne04_patches
python -m part2_classification.evaluate_dermnet   --weights outputs/classifier/best.pt --dermnet-root data/dermnet
python -m part2_classification.gradcam            --weights outputs/classifier/best.pt --predictions outputs/dermnet_eval/predictions.json --dermnet-train data/dermnet/train
```

## Key design choices

- **Three-detector comparison** spans all three modern detector families (one-stage CNN, two-stage CNN, transformer). Each is trained from a strong pretrained checkpoint to converge on ~1450 images.
- **Faster R-CNN** uses **shrunk anchors** `(4, 8, 16, 32, 64)` (vs. COCO defaults `32–512`) because acne lesions are typically 5–30 px.
- **Unified evaluation** via `pycocotools.COCOeval` so all three models report the same mAP/P/R/IoU numbers.
- **Patch generation** for Part 2: positives from GT bboxes expanded by 1.3×, negatives sampled with strict IoU=0 against all GT in the same image, both resized to 224.
- **Domain adaptation stack** (cheap to expensive): heavy training-time augmentation → Reinhard color normalization at test time using a 20-image DermNet reference mosaic → TTA.
- **Grad-CAM** picks a stratified mix of TP/FP/TN/FN predictions for a more diagnostic visualization than random sampling.

## Reproducibility

- Train/val/test splits come straight from Roboflow's ACNE04 export, no resplitting.
- Ultralytics seeds itself; for the Faster R-CNN loop set `torch.manual_seed(0)` before `train()` if you need exact repeats.
- All metrics/predictions are dumped to `outputs/` as JSON for downstream inspection.

## See also

- `report.md` — model selection rationale, preprocessing, training details, findings, and reflection on the domain gap.
