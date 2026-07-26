---
license: mit
tags:
  - ultralytics
  - yolo11
  - pose-estimation
  - keypoint-detection
  - wildlife
  - bird
  - eye-localization
library_name: ultralytics
base_model: ultralytics/yolo11n-pose
datasets:
  - synthet/image-scoring-model
---

# eye-pose-v0

YOLO11n-pose model fine-tuned on CUB-200-2011 for wildlife bird head localization.
Used as Stage-1 localization in the [image-scoring-model](https://github.com/synthet/image-scoring-model) pipeline for eye-focus scoring.

## Keypoints (6)

| Index | Name |
|------:|------|
| 0 | beak |
| 1 | left_eye |
| 2 | right_eye |
| 3 | head_top |
| 4 | left_shoulder |
| 5 | right_shoulder |

`flip_idx: [0, 2, 1, 3, 5, 4]`

## Training

- **Base:** YOLO11n-pose
- **Dataset:** CUB-200-2011 converted to YOLO pose format (~10k train / 1.7k val)
- **Epochs:** 100 (imgsz 640, batch 16)
- **Final validation (epoch 100):**
  - Pose mAP50: **0.994**
  - Pose mAP50-95: **0.983**
  - Box mAP50: **0.994**

## Usage

```python
from ultralytics import YOLO

model = YOLO("hf://synthet/eye-pose-v0/eye_pose_v0.pt")
results = model.predict("bird.jpg", imgsz=640)
```

Or with the `eye_quality` package:

```bash
pip install -e "git+https://github.com/synthet/image-scoring-model.git"
eye-quality score bird.jpg --weights eye_pose_v0.pt
```

Download weights:

```bash
huggingface-cli download synthet/eye-pose-v0 eye_pose_v0.pt --local-dir models/
```

Model files: [synthet/eye-pose-v0 on Hugging Face](https://huggingface.co/synthet/eye-pose-v0/tree/main)

## Limitations

- Single-class bird detection with head keypoints; not a general animal pose model.
- Trained on CUB-200 studio-style bird photos; performance on distant field wildlife may vary.
- Eye *localization* only — focus/sharpness scoring uses separate heuristics in the parent repo.
