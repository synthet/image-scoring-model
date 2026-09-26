# `training/` — train and convert

Scripts and Ultralytics configs for building datasets and fine-tuning pose/detect models.

| Path | Semantics |
|------|-----------|
| `convert_cub200.py` / prepare helpers | CUB-200 → YOLO labels under `data/` |
| `train_pose.py` | Fine-tune YOLO pose → `models/eye_pose_v0.pt` |
| `train_detect.py` | Fine-tune YOLO detect → `models/bird_detect_v0.pt` |
| `train_ctl.py` | Pause/resume helpers for long runs |
| `teacher/` | Upstream RTMDet-tiny (own PyTorch impl), ONNX export, pool selection, pseudo-labelling ([guide](../docs/guides/TEACHER_PSEUDO_LABELS.md)) |
| `configs/*.yaml` | Dataset YAMLs pointed at `data/` |

Guide: [`docs/guides/TRAINING.md`](../docs/guides/TRAINING.md).  
Before unattended runs: skill `autonomous-run-contract`.

Weights land in [`models/`](../models/README.md); logs in [`runs/`](../runs/INDEX.md). Never commit either.
