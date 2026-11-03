# `training/` — train and convert

Scripts and Ultralytics configs for building datasets and fine-tuning pose/detect models.

| Path | Semantics |
|------|-----------|
| `convert_cub200.py` / prepare helpers | CUB-200 → YOLO labels under `data/` |
| `train_pose.py` | Fine-tune YOLO pose → run-scoped candidate weights |
| `train_detect.py` | Fine-tune YOLO detect → run-scoped candidate weights |
| `run_safety.py` | Reject reused run directories and outputs outside candidate locations |
| `train_ctl.py` | Pause/resume helpers for long runs |
| `teacher/` | RTMDet pseudo-labels ([guide](../docs/guides/TEACHER_PSEUDO_LABELS.md)); `eye_alignment.py` / `rtmpose_bird_eye.py` for verified teacher→YOLO geometry (local ONNX path only) |
| `partial_pose.py` | Masked keypoint/objectness loss for partial teacher labels (not wired into `train_pose.py` yet) |
| `eval_field.py` / [FIELD_EVALUATION.md](FIELD_EVALUATION.md) | Ground-truth bird and eye benchmark, manifest contract, and metrics |
| `sample_field_eyes.py` / `label_eyes/` | Stratified read-only library sample (renditions + detector boxes) and a local click-labelling page for eyes |
| `eval_eye_benchmark.py` | Compare eye models (YOLO pose, production pipeline, two-stage crop, top-down SimCC ONNX by path) on CUB held-out species or a labelled field sample |
| `build_verified_negatives.py` | Import owner-reviewed bird-free photos into a private, hashed training manifest; reject holdout-folder overlap |
| `configs/*.yaml` | Dataset YAMLs pointed at `data/` |

Guide: [`docs/guides/TRAINING.md`](../docs/guides/TRAINING.md).  
Before unattended runs: skill `autonomous-run-contract`.

Weights land in [`models/`](../models/README.md); logs in [`runs/`](../runs/INDEX.md). Never commit either.
