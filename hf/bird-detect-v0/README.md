---
license: mit
tags:
  - ultralytics
  - yolo11
  - object-detection
  - wildlife
  - bird
library_name: ultralytics
base_model: ultralytics/yolo11n
datasets:
  - synthet/image-scoring-model
---

# bird-detect (v0, v1)

YOLO11n detect models for bird bounding boxes (single class `bird`).
Companion to [synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0) for subject localization
when eye keypoints are not required (species crops, gating, counting).

Used with the [image-scoring-model](https://github.com/synthet/image-scoring-model) `eye-quality detect` CLI.

| File | What | Use |
|---|---|---|
| `bird_detect_v1.pt` | **v1** (2026-09-27): v0 fine-tuned on CUB plus teacher pseudo-labels from real field photos | Recommended |
| `bird_detect_v0.pt` | v0: CUB-200-2011 only | Kept for reproducibility |

## Class

| Index | Name |
|------:|------|
| 0 | bird |

## v1

**Why.** v0 learned from CUB's well-framed, centred birds. On long-lens field frames it missed most small or
distant birds and had never seen a bird-free frame.

**Training.**
- **Start:** `bird_detect_v0.pt`, fine-tuned 30 epochs (imgsz 640, batch 16, lr0 0.002).
- **Data:** CUB-200-2011 boxes (10,018 train images) plus **803 teacher-labelled field frames** listed 5× per
  epoch (29% of each epoch):
  - 455 bird boxes and 385 bird-free frames;
  - labelled by the upstream Apache-2.0 **RTMDet-tiny COCO** detector used as a teacher (full frame plus 2×2
    tiles; only COCO `bird` ≥ 0.50 makes a box; anything ambiguous is excluded);
  - thresholds calibrated on an owner-labelled cohort whose folders are excluded from training.
- **Validation (CUB val + teacher val, final epoch):** mAP50 0.994, mAP50-95 0.878, precision 0.991,
  recall 0.982. Not comparable with v0's CUB-only validation numbers: the set now includes harder field frames.

**Field evaluation** (339-frame owner-labelled cohort from a wildlife library, never used in training;
presence per frame, Wilson 95% intervals):

| Frames | v0 | v1 |
|---|---|---|
| Birds v0 missed (78 bird frames): recall | 0% (0-5%) | **81% (71-88%)** |
| Bird-free frames among them (71): false positives | 0% (0-5%) | 7% (3-15%) |
| Frames v0 detected, with a bird (101): recall | 100% | 98% (93-99%) |
| Frames v0 detected wrongly (28 bird-free): still fire | 100% | **50% (33-67%)** |
| Small birds (< 4% of frame, 48): recall | 100% | 100% |

## v0 training

- **Base:** YOLO11n (`yolo11n.pt`)
- **Dataset:** CUB-200-2011 boxes via `data/wildlife_bird_det` (~10k train / 1.7k val)
- **Epochs:** 100 (imgsz 640, batch 16)
- **Final validation (epoch 100):** Box mAP50 0.994, mAP50-95 0.892, precision 0.993, recall 0.997

## Usage

```python
from ultralytics import YOLO

model = YOLO("hf://synthet/bird-detect-v0/bird_detect_v1.pt")
results = model.predict("bird.jpg", imgsz=640)
```

Or with the `eye_quality` package:

```bash
pip install -e "git+https://github.com/synthet/image-scoring-model.git"
huggingface-cli download synthet/bird-detect-v0 bird_detect_v1.pt --local-dir models/
python -m eye_quality detect bird.jpg --weights models/bird_detect_v1.pt
```

## Limitations

- Single-class bird boxes only; not a multi-species detector.
- v1's field data comes from one photographer's library (North American birds, long lens); validate on yours.
- The teacher excludes very small, silhouetted or occluded birds rather than labelling them, so those remain weak.
- CUB labels are typically one bird per image; crowded frames need care.
- CUB-200-2011 is a research dataset; check its terms for your use.
