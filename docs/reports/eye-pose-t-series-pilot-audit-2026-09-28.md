---
type: Report
title: T-series eye teacher pilot audit (200 frames)
description: Quality gate for ONNX bird-eye pseudo-labels before T0/T1 training.
resource: docs/reports/eye-pose-t-series-pilot-audit-2026-09-28.md
tags: [training, teacher-student, eye-pose, pseudo-labels, audit]
timestamp: 2026-09-28T23:46:00Z
okf_version: 0.1
---

# T-series eye teacher pilot audit

## Scope

Gate 0 provenance and training permission passed after the RTMDet adapter hash in
`models/rtmdet_tiny_coco.manifest.json` was updated to match the current NumPy-NMS module.
The new inference run used the first 200 entries of `data/pseudo_eye_v0/pool.json` and wrote
`data/pseudo_eye_pilot200/teacher_eye_detections.jsonl` plus preview cache. It did not
materialize labels or start training. The earlier 50-frame `data/pseudo_eye_v0/` pilot remains
untouched.

| Decision | Frames |
|----------|-------:|
| bird | 24 |
| negative (teacher decision, not human-verified) | 63 |
| uncertain_keyword_only | 62 |
| uncertain_low_conf | 30 |
| uncertain_mammal | 21 |
| decode/inference errors | 0 |

The 24 bird frames contain 28 accepted pose instances. Only four instances across three
images have any mapped eye keypoint. All four have both eyes marked observed.

## Visual quality gate

The four eye-labeled instances failed visual acceptance at the cached preview resolution:

- `235880`: two side-view doves, each with one visible eye; both received two observed-eye labels.
- `19899`: distant side-view wader; the second assigned eye is not resolvable in the preview.
- `205490`: side-view dove with one visible eye; the teacher assigned two observed eyes.

These are candidate detections, not approved training labels. This small reviewed set does not
estimate the error rate for all teacher outputs. Review original-resolution images and record
anatomical left/right and visibility before any eye pseudo-labels enter a training dataset.

The prior 50-frame pilot had seven bird instances with beak labels but zero known eye labels.
It also materialized 13 empty negative labels before the safety correction. Detector silence
and absent keywords do not verify bird absence; those legacy labels must not be used for
pose training. Preview inspection alone is insufficient to certify tiny or distant negatives.

## Current stop condition

T0/T1 training and the proposed 45/40/15 blend are on hold. The current field data has no
approved eye pseudo-labels or human-verified negatives. `training/partial_pose.py` is a loss
primitive; `training/train_pose.py` still uses the stock trainer and does not carry
`annotation_known` through augmentation, matching, and loss. At the initial audit, the Store Python
environment also failed to import the trainer because `torchvision::nms` was missing.

`pseudo_label_eye materialize` now treats teacher-negative decisions as audit-only and
rejects records with candidate eye labels before removing any existing materialized files.
The new 200-frame run remains detection-only. The focused materialization tests passed
(3 tests, with a temporary torchvision import shim); this does not exercise GPU training.

## Next reviewable steps

1. Collect human eye labels on original-resolution frames and establish anatomical
   identity for accepted points.
2. Build a versioned set of verified bird-free negatives, separate from teacher silence.
3. Implement and test the annotation-known mask through the installed Ultralytics data,
   augmentation, assignment, and pose loss path; verify the trainer runtime.
4. Prepare corrected CUB plus reviewed field blend, freeze independent evaluation, then run
   matched T0/T1 with run-scoped checkpoints and a training-run contract.

## Full-resolution follow-up (2026-09-29)

The largest embedded JPEGs from the three original RAW frames were reviewed without teacher
marks. They measured 5392×3592 (`235880`), 3936×2624 (`19899`), and 5392×3592 (`205490`).
Both doves in `235880`, the side-view bird in `19899`, and the dove in `205490` each show
one visible eye. All four proposed two-observed-eye labels are rejected. No anatomical
left/right eye label is approved from this pilot.

The `19899` teacher record was generated from a 160×120 RAW thumbnail: the shared teacher
decoder accepted Pillow's small RAW rendition before trying the full embedded JPEG. The
decoder now extracts the full embedded JPEG first for RAW input; a real decode of this frame
returns 3936×2624. This changes teacher inference input, so the 200-frame pilot counts are
historical and must not be treated as a calibrated result. At this follow-up, the local pilot
output and pool directories were absent from `data/`; a rerun needs a restored pool and a
fresh output directory.

The decoder regression failed before the change and passed after it. The focused RAW,
materialization, and alignment suite passed (15 tests). No GPU teacher inference or T-series
training was run in this follow-up. T0/T1 remain gated on reviewed eye labels, verified
negatives, the partial-keypoint trainer hook, and an independent field-eye evaluator.
