---
type: Technical Reference
title: Eye evidence as built
description: Localization, crop geometry, heuristic formulas and summary precedence reconstructed from code.
resource: specs/as-built/eye-evidence.md
tags: [specification, as-built, eye-quality]
timestamp: 2026-10-23T00:00:00Z
okf_version: 0.1
---

# Eye evidence as built

Observed baseline: `b80aa536c3597d9986ca3121fc0e6376741df57d`, with the working-file identities
frozen in the hub review dated 2026-10-23. This describes implemented behavior, including the review's
NumPy accumulation correction. It does not authorize weights promotion or promise natural-image accuracy.

## Architecture and inputs

`eye-quality score/batch` → [score_image](../../../src/eye_quality/pipeline.py) →
[load_oriented_image / crop_eye_region](../../../src/eye_quality/crop/eye_crop.py) →
[PoseLocalizer](../../../src/eye_quality/localization/pose_model.py) →
[compute_eye_metrics](../../../src/eye_quality/scoring/heuristics.py) →
[aggregate_eye_quality](../../../src/eye_quality/scoring/aggregate.py) → Pydantic JSON.
`localize_fn(path, oriented_width, oriented_height)` is the weight-free test seam. Ordinary file/decode/model
errors propagate from `score_image`; `empty_result` is a separate explicit no-op. Debug writes occur only
when `debug_dir` is supplied. Sources are read-only; output and batch error policy live in
[cli.py](../../../src/eye_quality/cli.py). JSON consumers use [API_CONTRACT](../../technical/API_CONTRACT.md).

Images are RGB after PIL EXIF transpose, with x increasing rightward and y downward. Pose boxes are pixel
xyxy; API eye boxes are normalized **xywh**, not xyxy. Neither representation may be silently substituted
for pipeline `display_normalized` xyxy regions. The detector currently reads untransposed file dimensions
before calling YOLO on the filename; actual YOLO orientation behavior requires a real-weight parity check.
The crop EXIF unit test alone does not establish end-to-end pose orientation parity.

## Detection and crop geometry

The six anatomical slots are `beak,left_eye,right_eye,head_top,left_shoulder,right_shoulder`;
horizontal flipping permutes `[0,2,1,3,5,4]` in [keypoint_schema](../../../src/eye_quality/localization/keypoint_schema.py).
Pose filtering uses bbox confidence ≥0.25 and area/frame area ≥0.002. Detections sort by bbox confidence;
only the first supplies eyes. Explicit weights win; otherwise `eye_pose_v0.pt` wins if present, else
`yolo11n-pose.pt` with a warning. A generic human pose checkpoint does not establish bird keypoint semantics.

Head width is `max(abs(right_eye.x-left_eye.x),16)` when both confidences are positive; otherwise
`max(abs(right_shoulder.x-left_shoulder.x)*0.35,16)` when both shoulders are positive; otherwise
`max(bbox_width*0.4,16)`. It is horizontal separation, not Euclidean/head-rotation-normalized distance;
`head_top` is used in turned-away logic, not in this width formula. A confident beak with no confident eyes
means turned away; with a confident head-top, beak y < average visible-eye y −5 also means turned away.

Visibility precedence: turned-away and confidence <0.45 → turned_away; confidence <0.25 → hidden;
crop-size check → too_small; out-of-image point → occluded with confidence×0.7; confidence ≥0.45 → visible;
otherwise partially_visible. The size check uses `2*max(head_width*0.12,12/2)`, so it cannot fall below
12 under the default formula. Hidden/too-small localizations with confidence <0.25 are omitted.

The localization's API bbox radius is `max(head_width*0.12,8)` and is clipped to the frame, with minimum
normalized width/height of one pixel. The **actual scoring crop** instead uses radius
`max(head_width*0.12,12)`, integer-rounded center/radius, and clipped native pixels. Thus reported bbox and
saved crop need not be identical. Invalid crop bounds return a 1×1 zero image and reported crop width 0.

## Exact scoring formulas

Let gray intensities be 0–255. OpenCV uses Laplacian variance and 3×3 Sobel magnitude mean; NumPy fallback
uses the interior five-point Laplacian and first-difference gradient magnitude. These are different kernels.
Clarity is whole-crop RMS about its mean (the name “local RMS” does not imply sliding windows).

| Signal | Implemented formula |
|---|---|
| sharpness S | clip(Laplacian variance / 500, 0, 1) |
| clarity C | clip(RMS / 64, 0, 1) |
| edge E | clip(mean gradient magnitude / 80, 0, 1) |
| noise N | clip(mean(abs(gray − blur)) / 25, 0, 1) |
| blur | OpenCV Gaussian sigma 1.2; fallback edge-padded 3×3 box mean |
| size Z, width w | 0 if w≤0; w/48 if 0<w<24; clip(w/48,0.5,1) otherwise |
| focus F | clip((0.45S+0.35C+0.20E) Z max(0,1−N),0,1) |

Fallback neighbors now accumulate in float before averaging; uint8 addition previously wrapped, producing
N=1 even on a constant crop at intensity 32,64,128 or255. This correction is independently regression-tested.
Empty or dimension<2 crops yield S=C=E=Z=0, N=1 and hidden_eye. Tiny 2-pixel dimensions in the non-OpenCV
Laplacian path remain an explicit numerical boundary gap; this review does not claim full backend parity.

Exposure precedence: mean>220 OR fraction(gray≥250)>0.15 → overexposed; else mean<35 OR
fraction(gray≤5)>0.25 → low_contrast; else none. This is exposure evidence, not a physical blur classifier.

## Aggregation and outputs

Exclude hidden, too_small and turned_away; partially_visible and occluded remain candidates. Sort by
`(focus_score,sharpness_score)` descending, preserving input order for exact ties. **Confidence is not a
preselection gate**. Pick the first; halve its confidence when crop width<24. At confidence<0.45, failure
is none and grade is capped at2; its focus score is unchanged. At confidence≥0.45: size_factor<0.4 wins,
then preexisting failure, then sharpness<0.15 AND focus<0.25 → defocus, then focus<0.20 → motion_blur,
then noise>0.6 → noise. The explicit turned-away/too-small branches are unreachable after visibility filtering.

Grades are 4 at F≥0.85, 3 at≥0.65, 2 at≥0.45, 1 at≥0.20, else0. No candidates produces count0,
null best side, scores/confidence/grade0 and failure none; detections/debug remain available. For example
F=0.90 with confidence0.30 remains F=0.90 but grade2/failure none, even if another eye has F=0.70/confidence0.9.
This observed confidence/quality separation is not a requirement that such a candidate should win.

## Acceptance and known limits

| Criterion | Regression evidence |
|---|---|
| Flat/noisy inputs, empty crop, grade and confidence boundaries | [test_heuristics](../../../tests/test_heuristics.py), [test_aggregate](../../../tests/test_aggregate.py) |
| Constant crop and brightness shift do not invent fallback noise | [test_heuristic_fallback](../../../tests/test_heuristic_fallback.py), five red→green cases |
| Crop EXIF transpose and debug side effects | [test_eye_crop](../../../tests/test_eye_crop.py) |
| Public ranges and empty pipeline | [test_schemas](../../../tests/test_schemas.py), [test_pipeline](../../../tests/test_pipeline.py) |
| Natural-image localization/focus accuracy, all EXIF/model paths, calibrated uncertainty | Gap: requires frozen group-separated field evaluation and weight identities |

Proposed head-relative measurements and confidence-gated candidate selection are experiment candidates,
not implemented policy. Keep technical eye evidence separate from duplicate equivalence and editorial fit.
