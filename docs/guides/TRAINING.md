---
type: Guide
title: Training
description: CUB-200 bootstrap, YOLO fine-tuning, resume, and validation for eye-quality.
resource: guides/TRAINING.md
tags: [docs, training, guide]
timestamp: 2026-07-26T00:00:00Z
okf_version: 0.1
---

# Training

Fine-tune a YOLO11n-pose model on CUB-200-2011 bird part annotations mapped to our 6-keypoint head schema.

## Prerequisites

- Python ≥ 3.10, package installed: `pip install -e ".[dev]"`
- CUDA-enabled PyTorch for GPU training:

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  python -c "import torch; print(torch.cuda.is_available())"
  ```

## Download pretrained weights

Skip training by downloading the published checkpoint from Hugging Face: **[synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0/tree/main)**

```bash
pip install huggingface_hub
huggingface-cli download synthet/eye-pose-v0 eye_pose_v0.pt --local-dir models/
```

See also [models/README.md](../../models/README.md). To reproduce or improve weights, follow the CUB-200 bootstrap below.

## CUB-200 bootstrap

### 1. Download dataset

[CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/) or the fast.ai mirror:

```bash
curl -L -o CUB_200_2011.tgz https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
```

### 2. Convert to YOLO pose format

`training/convert_cub200.py` maps CUB's 15 part landmarks to our schema. CUB provides separate `left eye` and `right eye` parts; only observed eye coordinates are retained. Head geometry is not treated as an eye label.

```bash
python -m training.convert_cub200 \
  --cub-root D:/Datasets/CUB_200_2011 \
  --output data/wildlife_bird \
  --val-ratio 0.15
```

Check the converter's reported counts for the current source and settings; birds without visible eyes retain their box targets.

Dataset layout:

```text
data/wildlife_bird/
  images/train/   labels/train/
  images/val/     labels/val/
```

Ultralytics config: `training/configs/wildlife_bird.yaml`.

### 3. Fine-tune

**Smoke test** (3 epochs, verifies GPU and data paths):

```bash
python training/train_pose.py --name cub_pose_smoke_1 --epochs 3 --batch 8 --device 0
```

**Full run** (copies best weights to `models/candidates/pose_cub_pose_1.pt` on completion):

```bash
python training/train_pose.py --name cub_pose_1 --epochs 100 --batch 16 --device 0
```

Training artifacts are written under `runs/pose/<name>/`. Every new run needs a unique name. Candidate weights are not committed to git; the release checkpoint is on [Hugging Face](https://huggingface.co/synthet/eye-pose-v0/tree/main).

### 4. Resume interrupted training

```bash
python training/train_pose.py \
  --name cub_pose_1 --resume runs/pose/cub_pose_1/weights/last.pt \
  --epochs 100 --batch 16 --device 0 \
  --output models/candidates/pose_cub_pose_1.pt
```

Ultralytics restores optimizer state and epoch counter from `last.pt`. The run continues in the same `save_dir` as the checkpoint.

### 5. Pause and resume

Use `training/train_ctl.py` to stop and restart long runs without hunting for PIDs or checkpoint paths:

```bash
# Show current run (PID, checkpoint, completed epochs)
python -m training.train_ctl status --task detect

# Pause the active detect or pose training process
python -m training.train_ctl pause --task detect

# Resume from runs/detect/wildlife_bird/weights/last.pt
python -m training.train_ctl resume --task detect --epochs 100 --batch 16 --device 0

# Start the controller's fixed-name run (auto-resumes if last.pt exists)
python -m training.train_ctl start --task detect --epochs 100 --batch 16 --device 0 --background
```

Pose training uses `--task pose` and writes checkpoints under `runs/pose/wildlife_bird/`.

State is recorded in `runs/.train_state.json`. On Windows, set `EYE_QUALITY_PYTHON` if your CUDA-enabled interpreter is not the default Store Python. Background runs append to `runs/<task>_train.log`.

**Resume is locked to the checkpoint's epoch count.** Ultralytics restores training args from `last.pt`, so `--epochs` on resume is ignored. A different budget needs a fresh run (new `--base`, or a different run `name`).

New direct-script runs reject an existing `runs/<task>/<name>/` directory. Resume only the matching `last.pt`; it may update that run's candidate output. `train_ctl` manages its fixed `wildlife_bird` run and writes candidate weights, so use the direct scripts with a unique `--name` for new comparisons. Released files under `models/` cannot be training outputs.

**Always launch from the repo root.** Dataset YAMLs use relative paths such as `path: data/wildlife_bird_det`. Ultralytics resolves those against the working directory; otherwise it falls through to `DATASETS_DIR` (`datasets/`). `train_ctl` already passes `cwd=REPO_ROOT`.

Manual resume (without `train_ctl`) still works:

```bash
python training/train_pose.py \
  --name cub_pose_1 --resume runs/pose/cub_pose_1/weights/last.pt \
  --epochs 100 --batch 16 --device 0 \
  --output models/candidates/pose_cub_pose_1.pt
```

### Detect-only bootstrap

Same CUB boxes without keypoints — for subject localization / BioCLIP crops when eye keypoints are not needed:

```bash
# 1. Pose dataset (if not already built)
python -m training.convert_cub200 \
  --cub-root D:/Datasets/CUB_200_2011 \
  --output data/wildlife_bird \
  --val-ratio 0.15

# 2. Strip keypoints → detect labels (hard-links images when possible)
python training/make_detection_labels.py \
  --pose-dir data/wildlife_bird \
  --output data/wildlife_bird_det

# 3. Fine-tune YOLO11n detect (run from repo root)
python training/train_detect.py --name cub_detect_1 --epochs 100 --batch 16 --device 0
```

Config: `training/configs/wildlife_bird_det.yaml`. This example writes artifacts under `runs/detect/cub_detect_1/` and candidate weights under `models/candidates/`.

Resume a paused detect run:

```bash
python training/train_detect.py \
  --name cub_detect_1 --resume runs/detect/cub_detect_1/weights/last.pt \
  --epochs 100 --batch 16 --device 0
```

For field improvement experiments, first freeze a reviewed field dataset and the
[ground-truth evaluation manifest](../../training/FIELD_EVALUATION.md). Use a new
dataset YAML and initialize from the published task-specific weights, with a new
run name for every controlled comparison:

```bash
python training/train_detect.py --data training/configs/<reviewed-field-detect>.yaml \
  --base models/bird_detect_v1.pt --name bird_field_d1 --epochs 40 --device 0
python training/train_pose.py --data training/configs/<reviewed-field-pose>.yaml \
  --base models/eye_pose_v0.pt --name eye_field_p1 --epochs 40 --device 0
```

These are templates, not runnable dataset paths. Do not treat teacher silence as a
verified negative or proposed eye points as human labels. Measure candidate
checkpoints on calibration and the untouched test set before changing any release
or backend selection.

### Owner-reviewed bird-free frames

The private owner cohort stores photo paths as `/mnt/d/Photos/...`; on Windows,
pass `D:/Photos` as `--photo-root`. The importer resolves and hashes the original
files, includes only explicit `no_bird` owner labels, and rejects any source
folder shared with the independent promotion holdout:

```powershell
python -m training.build_verified_negatives `
  --cohort D:/Projects/image-scoring-pipeline/.agent/scratch/bird_v1_owner_labels/cohort.csv `
  --labels D:/Projects/image-scoring-pipeline/.agent/scratch/bird_v1_owner_labels/labels.csv `
  --photo-root D:/Photos `
  --holdout-cohort D:/Projects/image-scoring-pipeline/.agent/scratch/bird_v1_promotion/validation/cohort.csv `
  --output data/verified_negatives_v1/manifest.jsonl
```

Use a new versioned output directory on rerun; the importer refuses to replace
an existing manifest. The result is source provenance, not a YOLO dataset: decode
photos into a consistent display rendition and pair these verified negatives
with reviewed positive bird boxes before training. Owner `bird` labels state
presence only and cannot be used as full box annotations. Keep the independent
promotion cohort out of training and model selection.

Inference (bbox JSON):

```bash
python -m eye_quality detect data/wildlife_bird_det/images/val/<image>.jpg \
  --weights models/candidates/detect_cub_detect_1.pt --device 0
```

## Validation

After fine-tuning, score sample images and inspect debug crops:

```bash
python -m eye_quality score \
  data/wildlife_bird/images/val/003.Sooty_Albatross__Sooty_Albatross_0001_1071.jpg \
  --debug-dir debug_crops
```

Check that:

- Eye keypoints land on the iris/pupil region
- Crops are sharp enough to distinguish focus differences within a burst
- Turned-away and tiny subjects get appropriate visibility/failure flags

CUB images are Flickr/studio-heavy. Validate on your own field wildlife bursts before setting culling thresholds.

## Keypoint label format

Each YOLO pose label line:

```text
<class> <cx> <cy> <w> <h> <kpt1_x> <kpt1_y> <kpt1_v> ... <kpt6_x> <kpt6_y> <kpt6_v>
```

Coordinates are normalized 0–1. Visibility `v`: 0 = absent, 1 = occluded, 2 = visible.

The leading `<class> <cx> <cy> <w> <h>` fields are the **bird bounding box**. The same CUB labels can train a detection-only YOLO model (box without keypoints). See [BIRD_DETECTION.md](../architecture/BIRD_DETECTION.md).

## Other data sources

| Source | Use |
|--------|-----|
| **3D-POP** | Supplemental pigeon keypoints |
| **Your wildlife library** | Target 500–2,000 labeled eye crops for MVP 2 learned classifier |
| **Custom COCO JSON** | `training/prepare_dataset.py --format coco` |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Invalid CUDA device` | Install CUDA PyTorch (see Prerequisites) |
| All images skipped in conversion | Ensure CUB `parts/parts.txt` lists `left eye` / `right eye` (not a single `eye` part) |
| Generic COCO keypoints at inference | Download [eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0/tree/main) to `models/eye_pose_v0.pt`; check CLI logs for weights path |
| OpenCV errors on Windows | Heuristic scoring falls back to NumPy; localization is unaffected |
| Background run vanished with no output | Check `runs/<task>_train.log`; confirm liveness via the process list and a `results.csv` tail — not the state file alone |
| `train_ctl status` raises `SystemError` / WinError 87 | Fixed in `_pid_alive` (Windows uses `tasklist`). On an old checkout, query the process list directly |
