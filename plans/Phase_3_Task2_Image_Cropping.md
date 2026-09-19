# Phase 3: Task 2 - Product Photo Cropping Engine

## Goals
Develop a smart 2:3 aspect ratio framing engine that crops product photos based on garment category (`TOPS`, `OUTER`, `BOTTOMS`, `DRESS`, `SET`) while preserving head, hem, and centering rules.

## Sub-Tasks
### 3.1 Keypoint & Body/Garment Detection
- Detect head/face, shoulders, waist, hem, and person boundaries using MediaPipe Pose / OpenCV segmentation.

### 3.2 2:3 Framing Rules Logic
- `TOPS` / `OUTER`: Head (~4% margin) to hem (cut legs/floor).
- `BOTTOMS`: Waistband to hem/feet.
- `DRESS`: Head to hem.
- `SET`: Head to lower piece hem.
- Enforce exact 2:3 aspect ratio (w:h), horizontal centering, and zero garment clipping.

### 3.3 CLI & Bounding Boxes CSV
- Build `crop.py <image_dir> --meta meta.csv --out boxes.csv`.
- Output `boxes.csv` (`filename,x,y,w,h`).

## Verification
- Automated validity check (Aspect ratio within 1% of 2:3, window inside image, face enclosed).
- Test on `images/train` (86 photos) and `images/dev` (30 photos).
