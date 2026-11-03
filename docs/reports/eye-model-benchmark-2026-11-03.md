---
type: Report
title: Eye model benchmark on CUB and a labelled library sample
description: Our YOLO eye models vs an external top-down SimCC bird-eye model; the two-stage crop is the measured win.
resource: docs/reports/eye-model-benchmark-2026-11-03.md
tags: [report, eye-pose, evaluation, benchmark]
timestamp: 2026-11-03T00:00:00Z
okf_version: 0.1
status: complete
---

# Eye model benchmark (2026-11-03)

We compared our eye localizers with an external top-down SimCC bird-eye model, an RTMPose-tiny with 3 points (two eyes and the beak) plus eye-state and visibility heads. The comparison used CUB held-out species and a library sample of 147 frames that the owner labelled. The external model is a private teacher: its identity and lineage are recorded only under gitignored `docs/private/` ([PRIVATE_LOCAL.md](../PRIVATE_LOCAL.md)).

## Findings

- **Eye placement:** `bird_pose_15_v0` places eyes best once a bird is found. On the 44 library birds that every model found, it gets 0.96 of eyes within 5% of the box diagonal (95% CI 0.90–1.00), against 0.81 for the external model and 0.83 for `eye_pose_v0`. On CUB, the external model gets about 0.48 of eyes within 2%, against 0.74–0.80 for ours.
- **Finding small birds:** run on the whole frame at 640 px, our pose models find only 21–30% of the small birds in the library. The external model found 59%, but on a box from our own `bird_detect_v1`. Its advantage comes from that two-stage design.
- **Two-stage crop (detector box → `bird_pose_15_v0` on the padded crop):**
  - library within 5% goes from 0.51 to 0.69 (CI 0.57–0.82);
  - small birds go from 0.32 to 0.68;
  - an eye is found in 0.91 of frames;
  - CUB is unchanged (0.960 vs 0.959).
- **Different blind spots:** the external model fails on dark, backlit and night birds; ours fail more on owls and other birds facing the camera.
- **False boxes:** `bird_detect_v1` at confidence 0.25 put boxes on half of the frames with no subject, and the two-stage path inherits them.

## Data and method

- **CUB:** each model was scored on its own held-out species. The `eye_pose_v0` and `bird_pose_15_v0` held-out sets do not overlap, so neither model is unseen on the other's set.
- **Library sample:**
  - 120 random bird frames from camera-original folders, 40 per size band (box area under 4%, 4–15%, over 15%), one per date folder;
  - plus 27 human-verified no-bird frames;
  - 2048 px renditions, labelled with `training/label_eyes/`.
- **Library labels:** the owner labelled every animal subject. Non-bird subjects (70 of 152) are tagged and excluded from the bird scores, which leaves 82 birds and 78 visible eyes.
- **Scoring** (`training/eval_eye_benchmark.py`):
  - eyes are matched to labels without left/right, and errors are normalised by the labelled box diagonal;
  - eye-confidence thresholds were frozen on CUB, not tuned on the library;
  - intervals are 95% bootstrap over frames.
- **Cross-check:** at confidence 0.5, `bird_pose_15_v0` scores 0.9575 within 5% here, against 0.9558 from `training.eval_bird15_parts`.

## Library, birds only

| Setup | Birds found | Within 2% | Within 5% | Within 10% | Frames with an eye found |
|---|---|---|---|---|---|
| `eye_pose_v0`, full frame | 0.60 | 0.22 | 0.44 | 0.51 | 0.68 |
| `bird_pose_15_v0`, full frame | 0.62 | 0.32 | 0.51 | 0.55 | 0.71 |
| External SimCC on `bird_detect_v1` box | 0.76 | 0.32 | 0.55 | 0.67 | 0.79 |
| **`bird_pose_15_v0` on `bird_detect_v1` crop** | 0.76 | **0.46** | **0.69** | **0.72** | **0.91** |

Thresholded scores. Small library samples give wide intervals (about ±0.12 on the within-5% column).

## Next steps

1. **Opt-in two-stage path in `PoseLocalizer`:** detector box → padded crop → pose → map back. Same output schema.
2. **`bird_pose_15_v0` as the default eye model:** this is an owner decision.
3. **Detector false positives:** calibrate the box threshold on a calibration split, not on this test sample.
4. **Library-domain fine-tuning:** owls, frontal heads and low light, on a separately labelled set. The 147-frame sample stays a test set.

## Reproduce

```powershell
python -m training.sample_field_eyes --root <library> --out data/field_eye_v1 --bodies "<camera folders>"
python -m training.label_eyes.serve data/field_eye_v1
python -m training.eval_eye_benchmark run --field data/field_eye_v1 --birds-only --thresholds <frozen.json> `
  --model "B=yolo,weights=models/bird_pose_15_v0.pt,eyes=6:10" `
  --model "B_td=topdown,weights=models/bird_pose_15_v0.pt,eyes=6:10" --out data/eye_benchmark/field
```
