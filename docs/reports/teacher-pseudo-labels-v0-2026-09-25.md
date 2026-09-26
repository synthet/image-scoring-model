---
type: Report
title: Teacher pseudo-labels v0 for bird_detect_v0
description: An upstream open COCO detector (RTMDet-tiny, Apache-2.0) reimplemented in plain PyTorch and used as a teacher to pseudo-label bird boxes and hard negatives from the library; policy calibrated on the #377 owner labels, 877 training images produced.
resource: docs/reports/teacher-pseudo-labels-v0-2026-09-25.md
tags: [report, detection, pseudo-labels, teacher-student, rtmdet, bird-detect]
timestamp: 2026-09-25T00:00:00Z
okf_version: 0.1
status: complete
---

# Teacher pseudo-labels v0 for `bird_detect_v0`

> **Provenance.** The teacher is the public OpenMMLab RTMDet-tiny COCO checkpoint (Apache-2.0). Every
> number below was measured on our own library. Thresholds are starting points to re-fit, not constants.

## Why

[BIRD_DETECTION.md](../architecture/BIRD_DETECTION.md) records the gap. On the backend #377 cohort,
`bird_detect_v0` (YOLO11n fine-tuned on CUB) found a bird on only a small share of the frames the owner
labelled as birds. An open COCO detector reached 82% recall at a 4% false-positive rate.

CUB teaches well-framed, centred birds. It never shows the student:
- small, distant, backlit or occluded birds in long-lens frames
- hard negatives: frames with no bird at all

A stronger, licence-clean detector can supply both kinds of example at the scale of the library.

## Teacher

| Item | Value |
|---|---|
| Model | RTMDet-tiny, COCO 80 classes (bird = class 14) |
| Weights | `rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth`, sha256 `78e30dcc…54a7fb` |
| Code | [`training/teacher/rtmdet_tiny.py`](../../training/teacher/rtmdet_tiny.py): our own torch + torchvision implementation. Parameter names match upstream, so the checkpoint loads with `strict=True`. No mmdet, mmcv or mmengine needed. |
| Parity | 2,638 of 2,638 detections (score ≥ 0.05, 8 frames) matched an independent ONNX export of the same checkpoint at IoU ≥ 0.99. Median box error 3×10⁻⁴ px, max score difference 4.5×10⁻⁵. |
| Our ONNX export | [`export_rtmdet_onnx.py`](../../training/teacher/export_rtmdet_onnx.py): raw per-prior outputs (no NMS in the graph), plus a `.manifest.json` with checkpoint, ONNX and module hashes. Export vs PyTorch: boxes ≤ 0.13 px, scores ≤ 1×10⁻⁴. |

Two details were needed to reach parity; both are easy to get wrong:
- **BatchNorm epsilon is 1e-5.** Assuming 1e-3 visibly breaks parity.
- **The final keep is 300 detections after NMS**, not 100. The other settings are 5,000 pre-NMS candidates,
  IoU 0.65, score threshold 0.001 and 200 per class.

## Pipeline

```text
select_pseudo_label_pool.py   read-only SQL → stratified pool.json (cohort images + folders excluded)
pseudo_label_rtmdet.py detect       decode → 2048 px → full frame + 2×2 tiles @ 60% → merge (NMS 0.5)
                                    → teacher_detections.jsonl + 1280 px cache (run once, resumable)
pseudo_label_rtmdet.py materialize  label policy → YOLO images/ labels/ {train,val} + manifest.json
```

**Pool strata.** At most 5 frames per folder and stratum:

| Stratum | Meaning | Selected |
|---|---|---|
| `miss_birdkw` | `bird_detect_v0` found nothing, but the frame has the `birds` keyword | 666 |
| `small_det` | `bird_detect_v0` box covers < 4% of the frame | 282 |
| `negative` | no `birds` / `wildlife` / `animals` keyword | 500 |
| `random` | everything else | 300 |

All 339 images of the #377 benchmark cohort **and all 132 folders they come from** are excluded, so the
benchmark stays a clean test.

**Tiling.** Birds under ~2% of the frame fall below what a 640 px input can resolve. The 2×2 tiles each
cover 60% of the frame. Tile boxes that touch an inner tile edge are dropped, because the neighbouring tile
or the full frame sees that bird whole.

## Label policy

| Decision | Rule | Training use |
|---|---|---|
| `bird` | COCO `bird` ≥ 0.50, box ≥ 0.05% of the frame | boxes (class 0) |
| `negative` | no animal box ≥ 0.35 and no animal keyword | empty label file |
| `uncertain_mammal` | any non-bird animal ≥ 0.50 | excluded |
| `uncertain_low_conf` | an animal box in [0.35, 0.50) not covered by an accepted box | excluded |
| `uncertain_keyword_only` | keyword says animal, teacher finds nothing confident | excluded |
| `uncertain_student_saw_bird` | would be negative, but `bird_detect_v0` itself found a bird (`small_det`) | excluded |

Accepted boxes that lie ≥ 70% inside a higher-scoring accepted box are dropped as tile/full-frame
duplicates.

The policy only accepts **clear** cases. A missed bird labelled "negative" teaches the student to ignore
birds, which does more damage than dropping an ambiguous frame.

### Calibration on the #377 cohort (evaluation only)

The teacher ran on the cohort's owner-labelled frames (238 bird, 99 no-bird). Those detections were used
only to choose thresholds and never entered the training data.

| bird ≥ | uncertain ≥ | Frames labelled bird | Of them no-bird (FP) |
|---|---|---|---|
| 0.40 | 0.30 | 151 | 5 |
| 0.40 | 0.35 | 188 | 6 |
| 0.45 | 0.35 | 164 | 4 |
| **0.50** | **0.35** | **151** | **3** |
| 0.40 | 0.40 | 229 | 12 |

**Chosen: 0.50 / 0.35.** About 2% of bird labels land on a no-bird frame, and no owner-labelled bird
frame became a negative.

### What visual QA caught

Sample sheets of the materialized boxes were checked by eye before the final run. They caught two bugs
that the counts alone would not have shown:

1. **Keyword-promoted mammals.** A first policy also accepted any confident COCO animal on frames tagged
   `birds`, because COCO sometimes calls raptors `bear`. The `birds` tag is itself a noisy CLIP keyword,
   so a rabbit and a cat were boxed as birds. Now only the COCO `bird` class makes a positive, and any
   confident non-bird animal marks the frame uncertain.
2. **Duplicate boxes.** A wader was boxed twice, once with and once without its reflection, from the tile
   and full-frame passes. The containment check above removes this.

After both fixes, a 9-frame sample was clean: a wren, songbirds, silhouettes on wires and branches, a
woodpecker, cardinals, and a small raptor on a light pole.

## Result

| Split | Images | Bird boxes | Negatives |
|---|---|---|---|
| train | 803 | 455 | 385 |
| val | 74 | 48 | 34 |

Split by folder hash (10% val), so no folder is in both.

| Stratum | bird | negative | excluded |
|---|---|---|---|
| `miss_birdkw` (666) | **160** | 0 | 506 |
| `small_det` (282) | 190 | 0 | 92 |
| `negative` (500) | 35 | **419** | 46 |
| `random` (300) | 73 | 0 | 227 |

- **160 positives are birds the student missed entirely**, and 190 more are small birds it found. These
  are the coverage gap.
- **419 hard negatives.** The student had never seen a bird-free frame before.
- **35 birds turned up in frames without any animal keyword.** Keywords are not a reliable negative filter
  on their own; the teacher is needed as a second check.
- **871 frames (50%) were excluded as uncertain.** That is the cost of a conservative policy. The
  largest group (297) is `birds`-tagged frames where the teacher found nothing confident, mostly tiny or
  heavily obscured subjects.

The data lives in `data/pseudo_rtmdet_v0/` (gitignored). Its `manifest.json` records the policy, the
per-stratum counts and the teacher manifest with its hashes.

## Limitations

- **The teacher has COCO's blind spots.** Very small, silhouetted or partly hidden birds end up excluded
  rather than labelled, so the student gets no positive signal for the hardest cases from this set.
- **Pilot scale.** 1,748 frames with a 5-per-folder cap. The same pipeline scales to the library.
- **The FP rate comes from 151 positives.** "About 2%" has a wide interval (3 of 151).
- **Box style differs.** COCO boxes include tails and extended wings, while CUB boxes are tight. Mixing
  the two may shift the student's box convention, so check IoU-based metrics against both.

## Next

1. Fine-tune `bird_detect_v0` on CUB plus this set. Re-run the #377 benchmark and report recall at a fixed
   false-positive rate against the teacher's 82% at 4%.
2. If recall improves without more false positives, scale the pool up and feed the same boxes to the
   pose model's detection stage and to the crop corpus in
   [ssl-landmark-pretraining.md](../planning/ssl-landmark-pretraining.md) (set U).
3. Keep the teacher only as a labeller. Runtime detection stays with the student, or with the backend
   cascade ([image-scoring-backend#408](https://github.com/synthet/image-scoring-backend/issues/408)).

## Related

- [guides/TEACHER_PSEUDO_LABELS.md](../guides/TEACHER_PSEUDO_LABELS.md): how to rerun
- [architecture/BIRD_DETECTION.md](../architecture/BIRD_DETECTION.md): detector gap and training priorities
- Backend: [upstream weight identity report](https://github.com/synthet/image-scoring-backend/blob/master/docs/reports/upstream-weights-identity-2026-09-25.md), [#426](https://github.com/synthet/image-scoring-backend/issues/426) keypoint and mask providers
