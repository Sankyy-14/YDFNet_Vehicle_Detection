# YDFNet — Vehicle Detection Module
### Vehicle Detection component of the AI Traffic Policy Generator

A from-scratch PyTorch implementation of the YDFNet architecture — **YOLOv10
backbone + BiFPN neck + DETR-style transformer detection head** — built,
trained, and evaluated on real UA-DETRAC traffic footage as the perception
layer for a larger AI-powered traffic policy generation system.

## Architecture

```
Image → YOLOv10 backbone (P3/P4/P5) → BiFPN (multi-scale fusion) →
DETR-style head (set prediction, no NMS) → class + bbox predictions
```

- **Backbone**: pretrained YOLOv10, multi-scale feature extraction
- **Neck**: custom Bidirectional Feature Pyramid Network (learned weighted fusion)
- **Head**: DETR-style transformer decoder with learned object queries,
  trained via Hungarian bipartite matching — no NMS post-processing needed
- **Loss**: Hungarian matcher + classification / L1 / GIoU box loss

## Results

Trained on 6,000 real images sampled from UA-DETRAC (82,085 total available),
12 epochs, RTX 4050 GPU (~69 min total training time).

Full results, methodology, and honest findings (including a documented
"query collapse" issue in later training epochs and a domain-gap finding on
regional vehicle types) are in
[`YDFNet_Vehicle_Detection_Results_Writeup.md`](./YDFNet_Vehicle_Detection_Results_Writeup.md).

**Best checkpoint**: `checkpoints/ydfnet_epoch5.pt`

## Project structure

```
ydfnet/
├── models/
│   ├── backbone.py       # YOLOv10 backbone wrapper (P3/P4/P5 extraction)
│   ├── bifpn.py           # BiFPN neck
│   ├── detr_head.py       # DETR-style transformer detection head
│   ├── loss.py             # Hungarian matcher + set-prediction loss
│   └── ydfnet.py           # Full assembled model
├── data/
│   ├── ua_detrac_yolo_dataset.py   # Real UA-DETRAC (YOLO-format) loader
│   ├── ua_detrac_dataset.py         # Real UA-DETRAC (official XML) loader
│   └── make_toy_dataset.py           # Synthetic data for pipeline testing
├── train.py                # Training loop
├── demo.py                  # Single-image inference + visualization
├── sweep_checkpoints.py     # Compare all checkpoints on one image
├── checkpoints/              # Saved model weights per epoch
└── outputs/                   # Demo/inference result images
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics scipy pillow numpy
```

## Training

```bash
python train.py --data-root "<path to UA-DETRAC DETRAC_Upload folder>" \
    --max-samples 6000 --epochs 12 --batch-size 8 --img-size 640 --device cuda
```

## Inference

```bash
python demo.py --checkpoint checkpoints\ydfnet_epoch5.pt --image <path.jpg> \
    --score-threshold 0.5
```

## Dataset

Trained on [UA-DETRAC](https://detrac-db.rit.albany.edu/) (YOLO-format export
via [Kaggle](https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset)).
Not included in this repo — see the dataset's own license/access terms.

## Part of a larger project

This module is the Vehicle Detection component of a six-person
**AI Traffic Policy Generator** — a decision-support system combining traffic
prediction (SUMO + XGBoost), vehicle detection (this module), and
multi-objective policy optimization (NSGA-II) for traffic management.