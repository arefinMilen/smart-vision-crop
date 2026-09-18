# Computer Vision Technical Suite - Master Execution Plan

## Project Overview
This repository contains a full-stack production-grade pipeline solving two key Computer Vision tasks:
1. **Task 1: HeadCut Detection** - Image quality assessment binary classification.
2. **Task 2: Dynamic Image Cropping** - Bounding box generation with intelligent subject-aware dynamic cropping (2:3 target aspect ratio).

## Master Execution Strategy & Directory Structure
```text
smart-vision-crop/
├── Candidate_Submission/
│   ├── Task1_Solution/
│   │   ├── predict.py
│   │   ├── predictions.csv
│   │   ├── requirements.txt
│   │   └── writeup.pdf
│   └── Task2_Solution/
│       ├── crop.py
│       ├── boxes.csv
│       ├── requirements.txt
│       └── writeup.pdf
├── plans/
│   ├── Master_Plan.md
│   ├── Phase_1_Setup.md
│   ├── Phase_2_Task1_HeadCut_Detection.md
│   ├── Phase_3_Task2_Image_Cropping.md
│   ├── Phase_4_Writeups.md
│   └── Phase_5_Testing_And_Packaging.md
├── Task_1_HeadCut_Detection/ (Dataset)
└── Task_2_Image_Cropping/ (Dataset)
```

---

## Phases Roadmap Summary

| Phase | Goal | Key Deliverables | Status |
|---|---|---|---|
| **Phase 1** | Workspace & Environment Setup | Directory layout, requirements.txt, data utils | `[Completed]` |
| **Phase 2** | Task 1: Head-Cut Detection | `predict.py` (Rule + Model), `predictions.csv` | `[Completed]` |
| **Phase 3** | Task 2: Product Image Cropping | `crop.py` (2:3 aspect framing), `boxes.csv` | `[Completed]` |
| **Phase 4** | Analysis & PDF Write-ups | `Task1_Solution/writeup.pdf`, `Task2_Solution/writeup.pdf` | `[Completed]` |
| **Phase 5** | Verification & ZIP Packaging | `Candidate_Submission.zip`, dry-run verification | `[Completed]` |
| **Phase 6** | Task 1 Model Optimization | Boost Dev Accuracy to 87.5%+, update `predict.py` | `[In Progress]` |
| **Phase 7** | Task 2 Dynamic Contour Engine | OpenCV contour/face fallback in `crop.py` | `[Pending]` |
| **Phase 8** | PDF Embedded Screenshots & Packaging | Visual chart diagrams, updated PDF write-ups & ZIP | `[Pending]` |
