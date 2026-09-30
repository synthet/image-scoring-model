# `training/` — train and convert

Scripts and Ultralytics configs for building datasets and fine-tuning pose/detect models.

| Path | Semantics |
|------|-----------|
| `convert_cub200.py` / prepare helpers | CUB-200 → YOLO labels under `data/` |
| `train_pose.py` | Fine-tune YOLO pose → `models/eye_pose_v0.pt` |
| `train_detect.py` | Fine-tune YOLO detect → `models/bird_detect_v0.pt` |
| `train_ctl.py` | Pause/resume helpers for long runs |
| `teacher/` | RTMDet pseudo-labels ([guide](../docs/guides/TEACHER_PSEUDO_LABELS.md)); `eye_alignment.py` / `rtmpose_bird_eye.py` for verified teacher→YOLO geometry (local ONNX path only) |
| `partial_pose.py` | Masked keypoint/objectness loss for partial teacher labels (not wired into `train_pose.py` yet) |
| `configs/*.yaml` | Dataset YAMLs pointed at `data/` |

Guide: [`docs/guides/TRAINING.md`](../docs/guides/TRAINING.md).  
Before unattended runs: skill `autonomous-run-contract`.

Weights land in [`models/`](../models/README.md); logs in [`runs/`](../runs/INDEX.md). Never commit either.
