# Phase 2: Task 1 - Head-Cut Detection

## Goals
Implement and compare both Rule-Based and Fine-Tuned Model approaches for detecting cut-off heads in product photos. Build CLI `predict.py` and output `predictions.csv`.

## Sub-Tasks
### 2.1 Rule-Based Approach
- Use OpenCV / Face Detection / Haar Cascades / Contour analysis to detect top-edge face clipping.
- Benchmark accuracy and false positives/negatives on `train` (200 images) and `dev` (40 images).

### 2.2 Fine-Tuned Model Approach
- Train a lightweight classifier (ResNet18 / MobileNetV3 / EfficientNet) using PyTorch with data augmentation.
- Evaluate train and dev accuracy, save best model weights.

### 2.3 CLI & Predictions CSV
- Build `predict.py <image_dir> --out predictions.csv` accepting single image directory.
- Generate valid `predictions.csv` for `train` and `dev` sets.

## Verification
- Test `predict.py` command on unseen directory structure.
- Verify CSV header (`filename,head_cut`) and 100% binary outputs (0 or 1).
