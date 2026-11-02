---
type: Technical Reference
title: Training and evaluation as built
description: Dataset conversion, teacher safeguards, candidate checkpoints and field evaluation boundaries.
resource: specs/as-built/training-and-evaluation.md
tags: [specification, as-built, training, evaluation]
timestamp: 2026-10-23T00:00:00Z
okf_version: 0.1
---

# Training and evaluation as built

Source baseline `b80aa536c3597d9986ca3121fc0e6376741df57d`; read the dated hub review's working-source
hashes for preexisting teacher/training edits. [TRAINING](../../guides/TRAINING.md) remains the operator
guide; this page records implementation and limits. No training or checkpoint promotion was performed.

## Data and model flow

[convert_cub200](../../../training/convert_cub200.py) converts source images/parts/boxes into YOLO pose
and detection labels. [prepare_dataset](../../../training/prepare_dataset.py) accepts COCO or CSV;
[make_detection_labels](../../../training/make_detection_labels.py) supplies detection labels.
YOLO pose labels have normalized box cx,cy,w,h followed by six x,y,visibility triples in
[keypoint_schema](../../../src/eye_quality/localization/keypoint_schema.py) order. Pixel coordinates
from teacher crops must first be inverted to the source frame; changing eye slot order changes supervision.
Dataset builders write derived files; they are not evidence of dataset rights or annotation correctness.

[build_verified_negatives](../../../training/build_verified_negatives.py) imports owner-reviewed bird-free
photographs into a private hashed manifest and rejects evaluation-folder overlap. Unknown bird presence
must not be used as a negative. Preserve labels, permission evidence, capture groups and source hashes
independently of experiment hyperparameters; a converted file's existence does not establish permission.

Teacher architecture:
RTMDet proposals → RAW/oriented crop → RTMPose bird-eye ONNX → calibrated point confidence/identity review →
crop-to-source transform → partial annotation mask → candidate training dataset.
[eye_alignment](../../../training/teacher/eye_alignment.py) provides `CropTransform.from_crop`,
`to_input`, `to_source`, `align_keypoints` and `PartialKeypoints.horizontal_flip`.
Transforms include crop origin, scaling and optional letterbox offset; x/y are independently inverted.
Anatomical mapping must be one-to-one, and `min_confidence` finite in[0,1]. `confirmed_absent` is explicit
annotation; low confidence, out-of-crop/padding and unknown identity stay unknown. Unknown and absent are
different states even when their serialized coordinates are zeros.

[rtmpose_bird_eye](../../../training/teacher/rtmpose_bird_eye.py) accepts a local ONNX file only. Its input
is RGB 256×256, ImageNet mean/std; SimCC outputs decode point maxima and visibility with calibration
`(0.6343,1.1351)`, plus orientation logits. Teacher eye0/eye1 identities are not automatically anatomical.
[test_eye_identity](../../../tests/test_eye_identity.py) demonstrates facing-camera suggestions and
abstention for a single confident eye. Suggestions require independent verification before supervision.
[test_teacher_raw_decode](../../../tests/test_teacher_raw_decode.py) and
[test_teacher_eye_alignment](../../../tests/test_teacher_eye_alignment.py) cover decode/coordinate handling.

## Partial supervision and checkpoint lifecycle

[masked_keypoint_losses](../../../training/partial_pose.py) accepts predictions/targets(N,K,3), Boolean
known mask(N,K), positive finite areas(N,1) and sigmas(K). Known visibility is0/1/2. Location loss uses
known visible points; objectness BCE indexes known points only. Unknown/absent coordinates are sanitized
before arithmetic (NaN×0 is still NaN). Unknown-only batches return differentiable zero losses. Box-only
instances do not dilute location loss. [test_partial_pose_supervision](../../../tests/test_partial_pose_supervision.py)
checks gradients and bad shapes. **This helper is not wired into the stock train_pose entry point**;
its presence does not mean the active trainer safely consumes partially annotated labels.

[train_pose](../../../training/train_pose.py) and [train_detect](../../../training/train_detect.py) default
to YOLO11n pretrained bases, 100epochs, imgsz640, batch16. Pose exposes cache none/ram/disk and Windows-specific
worker default2. [run_safety](../../../training/run_safety.py) rejects reused run directories, existing
candidate outputs, invalid names and paths outside the candidate destination. Resume must identify an
existing checkpoint. Successful training copies run `best.pt` into the run-scoped candidate output;
`--promote-only` copies an existing checkpoint into a fresh **candidate**, not production weights.
[test_training_run_safety](../../../tests/test_training_run_safety.py) covers overwrite prevention.

Checkpoint selection by trainer validation is not held-out field acceptance. Default detect resolution
tries `bird_detect_v1.pt`, then v0, and refuses a generic COCO fallback. Pose uses its separate fallback
chain. A training candidate is not automatically installed into either chain.

## Frozen field evaluation

[eval_field](../../../training/eval_field.py) evaluates precomputed predictions; it does not decode
images or run models. [FIELD_EVALUATION](../../../training/FIELD_EVALUATION.md) defines its JSONL contract.
Manifest schema_version1 requires image_id, path, group_id, calibration/test split, display_oriented
pixels, lowercase 64-character rendition SHA-256, positive integer dimensions and `reviewed=true`.
Every bird has a positive in-frame pixel xyxy bbox. Empty birds means reviewed bird-free, not unlabeled.
If eyes are supplied both anatomical sides need a state: visible/hidden/unresolvable/unknown; only visible
eyes have in-frame coordinates. Box-only records are excluded from eye metrics.

Groups and identical rendition hashes cannot cross splits. Prediction IDs must exactly cover manifest
IDs and match rendition hashes. Coordinates/confidences must be finite; scores lie in[0,1]. Bboxes are
matched one-to-one with maximum-cardinality IoU matching and stable IoU preference. Evaluation reports
detection, localization/eye coverage and error summaries; inspect CLI parameters for its frozen thresholds.
Missing prediction rows are errors, not silently removed from denominators.
[test_eval_field](../../../tests/test_eval_field.py) covers validation, matching, leakage and missingness.

The exploratory [eval_localization](../../../training/eval_localization.py) scans detections and confidence
statistics without equivalent human-ground-truth guarantees; it must not be reported as field accuracy.
Required campaign additions remain explicit: frozen natural-image prediction receipts, capture-group
uncertainty, subgroup counts, latency/memory and permission evidence. No synthetic-only metric establishes
natural-image accuracy, and teacher/judge labels must not be relabeled as human truth.
