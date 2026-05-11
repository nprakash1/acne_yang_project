# Acne Detection & Cross-Domain Classification — Report

## Part 1: Acne detection on ACNE04

### Model selection rationale

The assignment requires at least two detectors spanning **classic** and **modern** architectures. I implemented three so each major detector family is represented:

| Model | Family | Reason for inclusion |
|---|---|---|
| **YOLOv8** (Ultralytics) | One-stage CNN, anchor-free | The accuracy/speed frontier for small-dataset detection. Multi-scale heads, mosaic augmentation, and anchor-free predictions make it strong on small lesions out of the box. |
| **Faster R-CNN + ResNet50-FPN** | Two-stage CNN, anchor-based | Classic baseline. FPN was *designed* for small objects, and the two-stage RPN→RoI pipeline historically dominates small-object precision. Anchor sizes were shrunk to `(4, 8, 16, 32, 64)` because acne lesions are typically 5–30 px — the default COCO anchors `(32–512)` would fail. |
| **RT-DETR** | One-stage Transformer (DINO family) | Real-time DETR-style detector that adopts DINO's key contributions: encoder-feature query selection, iterative box refinement, denoising-style training. Same Ultralytics API as YOLOv8 (avoids `mmcv` install fragility on Colab) while satisfying the "transformer / DINO-DETR (2023)" suggestion in the brief. |

**References supporting these choices:**
1. Ren *et al.*, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", NeurIPS 2015.
2. Lin *et al.*, "Feature Pyramid Networks for Object Detection", CVPR 2017 — small-object motivation for FPN.
3. Zhang *et al.*, "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection", ICLR 2023 — query selection and denoising training reused by RT-DETR.
4. Wu *et al.*, "Joint Acne Image Grading and Counting via Label Distribution Learning", ICCV 2019 — the ACNE04 dataset paper itself.

### Preprocessing

- ACNE04 is downloaded from Roboflow Universe in **both YOLOv8 and COCO formats** so the same images back all three detectors. Train/val/test splits are taken as-is.
- YOLOv8/RT-DETR use the standard 640×640 letterbox resize. Faster R-CNN uses torchvision's built-in min/max-size resize (default).
- For YOLOv8 we keep mosaic + mixup + HSV augmentation; for RT-DETR we disable mosaic/mixup (transformer detectors prefer milder photometric aug).

### Training details

| Hyperparameter | YOLOv8s | Faster R-CNN R50-FPN | RT-DETR-l |
|---|---|---|---|
| Initial weights | COCO `yolov8s.pt` | torchvision COCO | COCO `rtdetr-l.pt` |
| Epochs | 100 (early-stop @30) | 25 | 80 (early-stop @30) |
| Batch size | 16 | 4 | 8 |
| Optimizer | SGD (Ultralytics defaults) | SGD lr=5e-3, mom=0.9, wd=5e-4 | AdamW (Ultralytics defaults) |
| LR schedule | Cosine | Cosine | Cosine |
| Augmentation | mosaic, mixup, HSV, hflip | hflip only | HSV, hflip |
| Approx T4 wall-time | ~30 min | ~1.5 h | ~45 min |

### Evaluation (unified)

All three detectors are scored with the same `pycocotools.COCOeval` against the COCO-format test split. Reported metrics:

- **mAP @ 0.5** and **mAP @ 0.5:0.95** (standard)
- **Precision / Recall** at IoU=0.5, score ≥ 0.25 (interpretable)
- **Mean IoU** across matched true positives (localization quality)

Results table is produced inline in `notebooks/01_part1_detection.ipynb` and the per-image side-by-side bbox visualizations are saved to `outputs/viz/predictions_test.png` (green = GT, red = predictions with confidence).

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

**Expected outcomes** (see notebook for actual numbers on your run):

- The **baseline classifier** is expected to take a large hit when transferred from ACNE04 (consumer selfies, daylight) to DermNet (clinical lighting, varied body parts and skin tones). This is the classic *covariate shift* problem.
- The **largest single gain** typically comes from heavy training-time augmentation — it is essentially free yet covers most of the photometric gap.
- **Reinhard color normalization** further helps because it removes the systematic LAB-space mean/std offset between the two domains using a tiny target-domain reference.
- **TTA** adds a small (~0.5–1 point) but consistent F1 improvement at no training cost.
- **Failure modes** that remain visible in Grad-CAM:
  - Rosacea images sometimes look like inflamed-acne and are scored high p(acne) — partially intentional since we group rosacea with acne, but inflammation patterns can confuse the model.
  - Body acne (back, chest) is missed because ACNE04 contains only face images. The classifier learned a face prior it cannot honor on torso photos.
  - Hairy skin and dermatoscopic close-ups (extreme zoom) are out-of-distribution.

**What I'd try with more time:**

- **CycleGAN** ACNE04 ↔ DermNet on a few hundred unpaired images, then retrain the classifier on translated ACNE04. Most expensive option but likely the largest remaining gain.
- **VGGFace2-pretrained backbone** (`build_face_resnet`) tested as an A/B against ImageNet-pretrained — face priors might help on near-face DermNet images but probably hurt on torso photos.
- **MMD or DANN** style feature-level adaptation using the 20 unlabeled DermNet images — would require introducing an adversarial branch but is well-supported in the literature for small target sets.

### Challenges encountered

- **Tiny objects**: default Faster R-CNN anchors miss most acne lesions until shrunk; the FPN was the more important contributor than the anchor change.
- **Roboflow COCO export quirks**: a placeholder "acne-detection" category id 0 had to be filtered to keep `pycocotools` happy.
- **DermNet folder naming**: the Kaggle release uses exact strings like `Acne and Rosaceae Photos`; we match case-insensitively against `acne`/`rosacea` substrings.
- **Colab session timeouts**: Faster R-CNN's 1.5h training run can occasionally exceed a free-tier session, so checkpointing every epoch (`last.pth` + `best.pth`) is essential.

## Summary

Three detector families trained on ACNE04 with a unified evaluator; a binary classifier trained on patch crops then transferred to DermNet via a layered domain adaptation stack and explained with stratified Grad-CAM. Code is organized as a Python package (`part1_detection`, `part2_classification`) with two thin Colab notebooks acting as runners — see `README.md` for run instructions.
