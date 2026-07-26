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

# bird-detect-v0

YOLO11n detect model fine-tuned on CUB-200-2011 bird bounding boxes (single class `bird`).
Companion to [synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0) for subject localization
when eye keypoints are not required (species crops, gating, counting).

Used with the [image-scoring-model](https://github.com/synthet/image-scoring-model) `eye-quality detect` CLI.

## Class

| Index | Name |
|------:|------|
| 0 | bird |

## Training

- **Base:** YOLO11n (`yolo11n.pt`)
- **Dataset:** CUB-200-2011 boxes via `data/wildlife_bird_det` (~10k train / 1.7k val)
- **Epochs:** 100 (imgsz 640, batch 16)
- **Final validation (epoch 100):**
  - Box mAP50: **0.994**
  - Box mAP50-95: **0.892**
  - Precision: **0.993**
  - Recall: **0.997**

## Usage

```python
from ultralytics import YOLO

model = YOLO("hf://synthet/bird-detect-v0/bird_detect_v0.pt")
results = model.predict("bird.jpg", imgsz=640)
```

Or with the `eye_quality` package:

```bash
pip install -e "git+https://github.com/synthet/image-scoring-model.git"
huggingface-cli download synthet/bird-detect-v0 bird_detect_v0.pt --local-dir models/
python -m eye_quality detect bird.jpg --weights models/bird_detect_v0.pt
```

## Limitations

- Single-class bird boxes only; not a multi-species detector.
- Trained on CUB-200 studio/Flickr-style photos; validate on your field library.
- CUB labels are typically one bird per image; crowded frames need care.
