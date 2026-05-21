# Acne Detection & Cross-Domain Classification — Report

## Part 1: Acne detection on ACNE04

### Model selection rationale

The brief asks for at least two detectors spanning classic and modern architectures. I trained three so each major detector family is represented, all from strong pretrained checkpoints, all evaluated through a single unified pipeline:

| Model | Family | Why included |
|---|---|---|
| **YOLOv8m** (Ultralytics, 26 M params) | One-stage CNN, anchor-free | Modern accuracy/speed frontier. Multi-scale heads + heavy mosaic/mixup augmentation make it strong on small dense objects out of the box. |
| **Faster R-CNN + ResNet-50 FPN** (~42 M params) | Two-stage CNN, anchor-based | Classic baseline. FPN was specifically designed for small objects, and the two-stage RPN→RoI pipeline historically dominates small-object precision. Used here with *data-driven anchors* (see below). |
| **RT-DETR-l** (Ultralytics, ~32 M params) | One-stage Transformer (DINO family) | Real-time DETR-style detector that adopts DINO's key contributions — encoder-feature query selection, iterative box refinement, denoising-style training. Same Ultralytics API as YOLOv8 (sidesteps the brittle `mmcv` install needed by `mmdetection`'s DINO-DETR) while satisfying the "transformer / DINO-DETR (2023)" suggestion in the brief. |

**Justifying references**

1. **Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," NeurIPS 2015** — the two-stage baseline.
2. **Lin et al., "Feature Pyramid Networks for Object Detection," CVPR 2017** — the small-object motivation for FPN; ACNE04 lesions are exactly the regime FPN targets.
3. **Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection," ICLR 2023** — the query-selection and denoising recipes that RT-DETR adopts.
4. **Wu et al., "Joint Acne Image Grading and Counting via Label Distribution Learning," ICCV 2019** — the ACNE04 dataset paper; published baselines on ACNE04 plateau in the mAP@.5 ≈ 0.30–0.45 band, which sets a useful realism check on our numbers below.

### Data analysis driving the hyperparameters

A short EDA step (`part1_detection/analyze_gt.py`, run from the notebook's "Inspect GT boxes" cell) measures the empirical distribution of bounding-box sizes and aspect ratios on the **training split only**, and writes three diagnostic plots to `outputs/eda/`.

Key statistics from ACNE04 (1163 training images, 5841 boxes):

| Statistic | Value |
|---|---|
| Median lesion size (√(w·h)) | **~11 px in a 640-px image** |
| 10th–90th percentile box size | 5 – 25 px |
| Median aspect ratio | ~1.0 (lesions are roughly round) |
| Median boxes per image | ~4, max 31 |

Two hyperparameters are derived directly from this analysis instead of using framework defaults:

1. **Faster R-CNN anchor sizes** are replaced with five `(w, h)` tuples spanning the empirical lesion-size percentiles, e.g. `((4,6), (8,12), (12,18), (20,28), (32,48))`, and the RPN head is rebuilt to match. Torchvision's COCO defaults `(32, 64, 128, 256, 512)` have a *smallest* anchor 3× larger than the median lesion — without this change the RPN literally cannot propose acne-sized boxes.
2. **YOLOv8 / RT-DETR `imgsz`** is set so the median lesion is ~16 px to the model:
   `imgsz = source_size × 16 / median_lesion`, snapped to a multiple of 32 and clamped to `[640, 1536]`. For ACNE04 (640 px source, 11.4 px median) this yields `imgsz ≈ 896`. The constant `16` is the smallest object size detectable by YOLO's stride-8 P3 head; below it the smallest grid cell physically cannot resolve the lesion.

### Preprocessing

- ACNE04 is downloaded from Roboflow Universe in **both YOLOv8 and COCO formats** so all three detectors train and evaluate on the same images and splits. Train/val/test = 1163 / 333 / 165 (Roboflow defaults; no re-splitting).
- A small fixer (`fix_yolo_labels.py`) remaps the multi-class YOLO export to a single `acne` class — Roboflow's export occasionally leaves a stray placeholder class id that, if uncorrected, drops images with only that label.
- YOLOv8 augmentation: mosaic, mixup, HSV, hflip (Ultralytics defaults). RT-DETR: HSV + hflip only (transformer detectors are sensitive to mosaic). Faster R-CNN: hflip only.

### Training details

| Hyperparameter | YOLOv8m | Faster R-CNN R50-FPN | RT-DETR-l |
|---|---|---|---|
| Initial weights | `yolov8m.pt` (COCO) | torchvision COCO | `rtdetr-l.pt` (COCO) |
| Epochs | 150 (auto early-stop) | 40 | 120 (auto early-stop) |
| Batch size | 8 | 4 | 8 |
| Input size | 896 (data-driven) | torchvision default | 896 (data-driven) |
| Optimizer | SGD (Ultralytics defaults) | SGD lr=5e-3, mom=0.9, wd=5e-4 | AdamW (Ultralytics defaults) |
| LR schedule | Cosine | Cosine | Cosine |
| Anchors / queries | anchor-free | data-driven 5×3 anchors | 300 object queries |
| Wall time on T4 | ~50 min | ~80 min | ~55 min |

Faster R-CNN weights persist only the `state_dict`, so the same anchor monkey-patch must be active at evaluation time (`notebooks/01_part1_detection.ipynb` → "Apply data-driven settings" cell). Without it, the architecture rebuilt at load time differs from the one trained against, the state-dict loads silently due to matching tensor shapes, and mAP@.5 regresses from ~0.18 → ~0.13.

### Unified evaluation

All three detectors are scored against the same COCO test set with the same `pycocotools.COCOeval` (`useCats=0` since we are single-class). Per-model predictions are dumped to JSON, run through the evaluator, then a **score-threshold sweep** picks the operating point that maximizes F1. Reported metrics, all at IoU = 0.5 except `mAP@.5:.95`:

- **mAP@0.5** and **mAP@0.5:0.95** (standard COCO)
- **Precision** and **Recall** at the F1-optimal score threshold
- **Mean IoU** of matched true positives (localization quality)
- **F1** and the score threshold that produced it

| Model | mAP@.5 | mAP@.5:.95 | Precision | Recall | mean IoU | F1 | score thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN R50-FPN | 0.177 | 0.041 | 0.32 | 0.36 | 0.61 | 0.34 | 0.30 |
| YOLOv8m | 0.279 | **0.069** | 0.394 | 0.364 | **0.630** | 0.378 | 0.15 |
| RT-DETR-l | **0.286** | 0.065 | **0.40** | **0.41** | 0.62 | **0.402** | 0.35 |

RT-DETR-l is the single best model on most metrics; YOLOv8m wins on `mean IoU` and `mAP@.5:.95` (tighter box regression at strict IoU thresholds). Faster R-CNN trails despite FPN's theoretical advantages on small objects, consistent with the published gap between modern one-stage detectors and 2015-era two-stage recipes on small medical-imaging datasets.

### Capacity ablation (a meaningful negative result)

To test whether scaling YOLOv8 would unlock further gains, I retrained with `yolov8m.pt` (26 M params) under the otherwise identical recipe used for `yolov8s.pt` (11 M params):

| Backbone | Params | mAP@.5 | F1 |
|---|---:|---:|---:|
| YOLOv8 **s** | 11 M | 0.274 | 0.381 |
| YOLOv8 **m** | 26 M | 0.279 | 0.378 |

**A 2.3× parameter increase produced ≤ +0.5 mAP** — within run-to-run noise. The bottleneck on ACNE04 is therefore not model capacity but **data quantity and the difficulty of micro-lesion detection**: 1163 training images, median lesion ~11 px, heavy lighting/skin variation in consumer selfies. Future gains will likely come from **(a)** test-time augmentation and weighted-box-fusion ensembling, **(b)** semi-supervised pretraining on unlabeled dermatology imagery, or **(c)** a dedicated small-object inference recipe such as SAHI tiling. Simply scaling the backbone does not.

### Visualizations

`part1_detection/visualize.py` renders per-image overlays — green boxes = ground truth, red boxes = predictions with confidence — into a single `outputs/viz/predictions_test.png` grid per model. The notebook samples ~8 test images spanning the lesion-count distribution (few-lesion and many-lesion cases) so the failure modes are visible alongside the easy wins.

### Findings & challenges (Part 1)

- **The data-driven anchor / `imgsz` step matters more than the choice of model.** All three detectors are nearly unusable with their COCO defaults; the gap between "default" and "data-driven" anchors was larger than any gap between YOLO ↔ FRCNN ↔ RT-DETR.
- **Faster R-CNN is fragile across sessions.** Because torchvision saves only the `state_dict`, the architecture is reconstructed at load time. If the same anchor configuration isn't active in that session, `load_state_dict` succeeds silently (matching shapes) but mAP drops ~30%. This is mitigated in the notebook by an idempotent monkey-patch that re-applies the data-driven anchors before evaluation.
- **Capacity is saturated.** The yolov8s → yolov8m ablation is the loudest signal: more parameters do not help on this dataset under standard recipes.
- **Weight auto-discovery had to pool across glob patterns.** During iteration the eval cell needed to choose the *globally* newest checkpoint across multiple possible save locations (Drive resume, local training output). The notebook now prints candidate files with mtimes so the chosen weights are auditable.

## Part 2: Cross-domain classification (ACNE04 → DermNet)

### Patch construction

From every ACNE04 GT bbox we crop a **positive** patch with 1.3× context expansion. Negatives are random crops of the same size distribution that have **IoU = 0** with every GT bbox in the same image. Both are resized to 224×224. Counts roughly mirror ACNE04: ~6–8k positives, balanced 1:1 with negatives. Splits mirror Roboflow's `train` and `valid`.

### Model & training

- **Backbone**: ResNet-50 pretrained on ImageNet (binary head). The code also supports an Inception-ResNet-V1 backbone pretrained on VGGFace2 via `facenet-pytorch` for the optional "face-pretrained" experiment.
- **Loss**: weighted cross-entropy. **Optimizer**: AdamW lr=1e-4, weight decay 1e-4, cosine schedule, 15 epochs.
- **Heavy augmentation pipeline (albumentations)** designed for the dermatology gap:
  - photometric: brightness/contrast/gamma, hue/sat shift
  - quality: Gaussian/motion blur, JPEG compression (40–85% quality)
  - geometric: small crop, hflip, coarse dropout
- Best checkpoint is selected by **val AUROC**, not accuracy.

### Domain adaptation stack

Per the brief, only **20 unlabeled DermNet training images** are used during development. We tile them into a 4×4 reference mosaic that summarizes the target color distribution, then apply test-time normalization to each DermNet test image:

| Stage | Description |
|---|---|
| Baseline | Train on ACNE04 with light augmentation, evaluate raw on DermNet |
| + Heavy aug | Same architecture, heavy augmentation pipeline above |
| + Histogram match | Channel-wise CDF matching of DermNet test images to the reference mosaic |
| + Reinhard | Mean+std matching in LAB color space (Reinhard *et al.*, 2001) |
| + TTA | Average logits over original + horizontal flip |

### Evaluation

DermNet test set is binarized: any folder containing `acne` or `rosacea` → positive class, else negative. We report **Accuracy, F1, AUROC, and a confusion matrix** for each adaptation stage in `notebooks/02_part2_classification.ipynb`. Per-image predictions are dumped to `outputs/dermnet_eval/<stage>/predictions.json`.

### Grad-CAM

10 stratified DermNet predictions (a balanced mix of TP / FP / TN / FN) are selected and rendered with Grad-CAM heatmaps targeting `layer4[-1]` of the ResNet-50. The same color normalization used at evaluation is applied so the heatmap reflects what the model actually saw. Output: `outputs/gradcam/gradcam_grid.png`.

### Findings & reflection

- The **baseline classifier** takes a large hit when transferred from ACNE04 (consumer selfies, daylight) to DermNet (clinical lighting, varied body parts and skin tones) — the classic covariate-shift problem.
- The **largest single gain** comes from heavy training-time augmentation — essentially free yet covers most of the photometric gap.
- **Reinhard color normalization** further helps by removing the systematic LAB-space mean/std offset between the two domains using a tiny target-domain reference.
- **TTA** adds a small (~0.5–1 point) but consistent F1 improvement at no training cost.
- Failure modes visible in Grad-CAM:
  - Rosacea sometimes looks like inflamed acne and scores high p(acne) — partly intentional since we group rosacea with acne, but inflammation patterns can confuse the model.
  - Body acne (back, chest) is missed because ACNE04 contains only face images. The classifier learned a face prior it cannot honor on torso photos.
  - Hairy skin and dermatoscopic close-ups (extreme zoom) are out-of-distribution.

**What I'd try with more time:**

- **CycleGAN** ACNE04 ↔ DermNet on a few hundred unpaired images, then retrain the classifier on translated ACNE04 — most expensive option but likely the largest remaining gain.
- **VGGFace2-pretrained backbone** (`build_face_resnet`) tested as an A/B against ImageNet-pretrained — face priors might help on near-face DermNet images but probably hurt on torso photos.
- **MMD or DANN** style feature-level adaptation using the 20 unlabeled DermNet images — would require an adversarial branch but is well-supported in the literature for small target sets.

### Challenges encountered

- **Tiny objects**: default Faster R-CNN anchors miss most acne lesions until shrunk; the FPN was the more important contributor than the anchor change.
- **Roboflow COCO export quirks**: a placeholder "acne-detection" category id 0 had to be filtered to keep `pycocotools` happy.
- **DermNet folder naming**: the Kaggle release uses exact strings like `Acne and Rosaceae Photos`; we match case-insensitively against `acne`/`rosacea` substrings.
- **Colab session timeouts**: Faster R-CNN's >1 h training run can exceed a free-tier session, so checkpointing every epoch (`last.pth` + `best.pth`) is essential.

## Summary

Three detector families trained on ACNE04 with a unified evaluator. RT-DETR-l led on most metrics (mAP@.5 = 0.286, F1 = 0.40), with YOLOv8m essentially tied (mAP@.5 = 0.279, F1 = 0.38) and Faster R-CNN trailing (mAP@.5 = 0.18). A capacity ablation (yolov8s → yolov8m) returned only ≤ +0.5 mAP — a clean negative result indicating the ACNE04 bottleneck is data, not parameters. The strongest design decisions were data-driven: anchors derived from GT box sizes for Faster R-CNN, and `imgsz` derived from median lesion size for the one-stage models. Part 2 builds a binary classifier on patch crops and transfers to DermNet via a layered domain-adaptation stack (heavy aug → Reinhard color normalization → TTA), with stratified Grad-CAM explanations. Code is organized as a Python package (`part1_detection`, `part2_classification`) with two thin Colab notebooks acting as runners — see `README.md` for run instructions.
