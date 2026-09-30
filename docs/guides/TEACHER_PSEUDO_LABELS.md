---
type: Guide
title: Teacher pseudo-labelling (RTMDet-tiny → bird_detect_v0)
description: How to fetch and verify the upstream RTMDet-tiny checkpoint, export it to ONNX, select a pool, run the teacher and materialize a YOLO pseudo-label dataset.
resource: docs/guides/TEACHER_PSEUDO_LABELS.md
tags: [guide, training, pseudo-labels, rtmdet, bird-detect]
timestamp: 2026-09-25T00:00:00Z
okf_version: 0.1
---

# Teacher pseudo-labelling

Code: [`training/teacher/`](../../training/teacher/). Results and rationale:
[teacher-pseudo-labels-v0-2026-09-25.md](../reports/teacher-pseudo-labels-v0-2026-09-25.md).

## 1. Fetch and verify the checkpoint

```bash
curl -L -o models/rtmdet_tiny_coco.pth \
  https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth
sha256sum models/rtmdet_tiny_coco.pth   # must start with 78e30dcc
```

`rtmdet_tiny.load_upstream()` loads the checkpoint without the OpenMMLab stack, by stubbing its metadata
classes during unpickling. The state dict must load with `strict=True`. If it doesn't, the file is not the
expected checkpoint.

## 2. Export to ONNX (optional, for runtimes other than PyTorch)

```bash
python -m training.teacher.export_rtmdet_onnx --ckpt models/rtmdet_tiny_coco.pth --out models/rtmdet_tiny_coco.onnx
```

- The exporter refuses a checkpoint whose hash does not match.
- It writes `models/rtmdet_tiny_coco.manifest.json`, which records the licence, the upstream URL, the
  checkpoint, ONNX and module hashes, and the input and post-processing settings.
- Outputs are raw per-prior values, `boxes (1,8400,4)` and `scores (1,8400,80)`. The runtime applies NMS
  itself, using the manifest's `postprocess` values to reproduce PyTorch.

`torch.onnx.export` needs the `onnx` package. If the GPU container lacks it, use a side venv rather than
changing the shared environment:

```bash
python -m venv --system-site-packages /tmp/onnxenv && /tmp/onnxenv/bin/pip install onnx
/tmp/onnxenv/bin/python -m training.teacher.export_rtmdet_onnx ...
```

## 3. Select the pool (read-only)

```bash
python -m training.teacher.select_pseudo_label_pool --out data/pseudo_rtmdet_v0/pool.json \
  --exclude-cohort ../image-scoring-backend/docs/reports/detector-benchmark-2026-09/cohort.csv \
  --per-folder 5 --seed teacher-v0
```

- The session is read-only.
- Every cohort image **and every cohort folder** is excluded.
- The four stratum sizes are set in `SIZES`.

## 4. Run the teacher once

```bash
python -m training.teacher.pseudo_label_rtmdet detect --pool data/pseudo_rtmdet_v0/pool.json \
  --ckpt models/rtmdet_tiny_coco.pth --out data/pseudo_rtmdet_v0
```

- Resumable: image ids already in `teacher_detections.jsonl` are skipped.
- Decoding uses the backend's `open_rendition_for_ml` + `bake_orientation` when it can be imported, so run
  it inside the backend GPU container. Otherwise it falls back to PIL with EXIF orientation.
- Takes about 1–2 s per frame on an 8 GB laptop GPU (5 passes at 640 px, plus the RAW decode).

## 5. Materialize with a policy

```bash
python -m training.teacher.pseudo_label_rtmdet materialize --out data/pseudo_rtmdet_v0 \
  --bird-thr 0.50 --uncertain-thr 0.35 --min-area 0.0005
```

- Rebuilds `images/` and `labels/` from scratch. Change thresholds and rerun freely.
- **Calibrate thresholds on a labelled hold-out, not by eye.** The v0 values were swept on the #377 owner
  labels. The same `detect` step can run over a cohort pool for that purpose. Keep that output outside the
  training directory.
- **Always check a rendered sample of accepted boxes across strata before training.** In v0 this caught
  mammals boxed as birds and duplicate boxes.

## 6. Train

Point a dataset YAML at `data/pseudo_rtmdet_v0/images/{train,val}`, alongside the CUB detect set
(`names: {0: bird}`), and fine-tune as in [TRAINING.md](TRAINING.md). Evaluate on the #377 cohort, which
this pool never touches.

## Rules

- The teacher is an **open, licensed** model (Apache-2.0), verified by hash.
- Never use proprietary or unlicensed models' outputs as labels. They may be run only as local evaluation
  baselines.
- Evaluation cohorts stay out of the pool, at folder level.

## Gate 0 — T-series eye pose (open det + optional proprietary pose)

Before generating **training** pseudo-labels for the optional eye-pose distillation branch (T-series), record
the **open** detector teacher in git and keep proprietary pose-teacher identity off git
([PRIVATE_LOCAL.md](../PRIVATE_LOCAL.md)).

| Teacher | Manifest | Gate report |
|---------|----------|---------------|
| RTMDet-tiny COCO (det) | `models/rtmdet_tiny_coco.manifest.json` | [teacher-gate0-t-series-2026-09-28.md](../reports/teacher-gate0-t-series-2026-09-28.md) |
| Proprietary bird-eye pose (local) | `docs/private/` (gitignored) | same report (open section only) |

Verify the open detector (and proprietary pose checks when your private manifest exists):

```bash
python -m training.teacher.verify_teacher_provenance --gate gate0 --allow-pending-permission
```

Omit `--allow-pending-permission` when the private manifest records granted training permission.

## Eye teacher (T-series pose pseudo-labels)

After Gate 0 passes, use the existing pool and a distinct pilot directory for audit. Pass a **local**
proprietary ONNX path (never committed):

```bash
python -m training.teacher.pseudo_label_eye detect --pool data/pseudo_eye_v0/pool.json \
  --eye-onnx <local-proprietary-onnx> --out data/pseudo_eye_pilot200 --limit 200
```

- Detection writes `teacher_eye_detections.jsonl` and preview cache.
- Materialization excludes teacher-negative decisions and rejects candidate eye labels. A reviewed-eye import path is still needed. The earlier `data/pseudo_eye_v0/` has 13 legacy unverified negatives; do not use it for training.
- The 200-frame pilot failed eye-label visual QA. See [pilot audit](../reports/eye-pose-t-series-pilot-audit-2026-09-28.md).
- Dataset YAML for the field slice alone: `training/configs/wildlife_bird_distill_v0.yaml`.
- Hold the corrected-CUB blend and T1 training until reviewed eye/negative labels and the partial-loss trainer hook exist.
