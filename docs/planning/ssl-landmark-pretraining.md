---
type: Plan
title: Self-supervised landmark pre-training on our own bird crops
description: Plan to pre-train a keypoint backbone on unlabelled bird crops from our library (invariant + equivariant contrastive learning, reimplemented from the published method), fine-tune on CUB plus our labels as a top-down eye/head model, and evaluate against eye-pose-v0.
resource: docs/planning/ssl-landmark-pretraining.md
tags: [planning, keypoints, eye-quality, self-supervised, training, cub]
timestamp: 2026-09-25T00:00:00Z
okf_version: 0.1
status: proposed
---

# Self-supervised landmark pre-training on our own bird crops

> **Status:** proposal. No code yet. It complements
> [eye-evidence-spec.md](../architecture/eye-evidence-spec.md) (MVP-2) and the backend's keypoint provider
> ([image-scoring-backend#426](https://github.com/synthet/image-scoring-backend/issues/426)).

## Why

The current `eye-pose-v0` (YOLO11n-pose fine-tuned on CUB-200-2011) localizes eyes well once it has a
bird. On the #377 cohort, where eye points from two models were compared on the same bird, the median
gap was 2.8% of the box diagonal. Its weaknesses are elsewhere:
- **Coverage:** it found a bird on only 147 of 238 owner-labelled bird frames, so the detection stage
  is the bottleneck.
- **Domain:** CUB photos are well-framed and centred. Our frames are long-lens, often small, backlit,
  partly occluded or in flight.
- **Confidence:** its eye confidences are high (median 0.98) and only loosely rank frames (ρ = 0.56
  against a reference model), so they need calibrating.

We have far more **unlabelled** bird crops from our own shooting conditions than labelled ones:
- about 41k legacy `bird_bbox` boxes in the backend
- more from the RTMDet teacher pseudo-labels (`training/teacher/`)

The method below learns landmark-aware features from such crops without labels. A few labels then
suffice to fine-tune.

## Method (reimplemented from the paper, not copied)

Source: Cheng, Su, Maji, *On Equivariant and Invariant Learning of Object Landmark Representations*,
ICCV 2021 ([arXiv:2006.14787](https://arxiv.org/abs/2006.14787)).

**Licence boundary:**
- The authors' repository has **no licence**. We implement the method from the paper and do not copy
  its code.
- Its released weights and CUB regressor may be run **locally for evaluation only** (§Evaluation).
  Their outputs never enter training data unless the authors grant permission.
- We don't use iNaturalist 2017 images.

**The two stages:**
1. **Invariant stage.** Contrastive learning (MoCo-style: momentum encoder plus a negative queue) on
   bird crops. The training signal is that two random augmentations of the same crop match, while
   different crops do not.
   - Augmentations: random resized crop within the padded box, flips, colour jitter, grayscale.
   - **No blur augmentation.** Blur is our quality signal, and this backbone should stay
     blur-sensitive for later reuse.
2. **Equivariant stage.** A light projector on **hypercolumn** features (concatenated upsampled
   features from several backbone stages) is trained so that a pixel's embedding follows known
   geometric warps.
   - Warp an image with a random thin-plate-spline or affine transform. Features at corresponding
     pixels should match, via a softmax matching loss with a temperature.
   - This teaches "the same part of the bird" across pose and shape.
3. **Fine-tuning.** A top-down keypoint head (heatmap or SimCC) on the hypercolumns predicts our
   6-point schema on crops: beak, left/right eye, head top, left/right shoulder.
   - Train on CUB (via the existing `training/convert_cub200.py` mapping) plus our own labels.

**Why top-down.** Detection and keypoints are separate problems here. The backend cascade (YOLO →
COCO → refine,
[#408](https://github.com/synthet/image-scoring-backend/issues/408)) supplies the box, and this
model only places points inside a padded crop at 192–256 px. That fixes the coverage gap without
retraining the keypoint model's detector, and matches the backend's region-crop design (#426).

## Data

| Set | Source | Use | Notes |
|---|---|---|---|
| **U** unlabelled crops | backend `bird_bbox` boxes (~41k) plus teacher boxes, cut from the ~2048 px rendition with 25% padding | stages 1–2 | Keep at most **k frames per continuous burst** (k = 3 to start). Near-identical burst frames otherwise dominate the negatives. |
| **C** CUB-200-2011 | parts mapped to the 6-point schema | fine-tuning, CUB test PCK | Research licence; check before redistributing weights (as today). |
| **L** our labels | agent-panel-verified eye/beak points, plus a small owner-checked set | fine-tuning, domain test | Target ~500 crops, stratified by subject size. Panel-proposed points are verified, never taken blind. |
| **H** held out | all frames from the #377 cohort's 132 folders | domain evaluation only | Never in U, C or L. Same exclusion as the teacher pool. |

Folder-grouped splits throughout. No folder appears in two splits.

## Model and compute (RTX 4060 laptop, 8 GB)

- **Backbone:** ResNet-18 first, cheap and the paper's small variant. ResNet-50 only if the gain
  justifies it.
- **Input:** 128–192 px for stages 1–2 (the paper used 96), 256 px for fine-tuning.
- **Rough budget, to be replaced by measurements:**
  - ~15–25k crops after burst deduplication
  - stage 1: ~200 epochs, batch 128
  - stage 2: ~20 epochs
  - on this card that is on the order of a day of GPU time
- **Baselines** at every label budget:
  - (a) ImageNet-initialized backbone
  - (b) current `eye-pose-v0`

## Evaluation

| Metric | Data | Question |
|---|---|---|
| PCK@0.05 / 0.1 of box size, per keypoint (eyes, beak) | CUB test | Is the new model at least as good as `eye-pose-v0` in-distribution? |
| **Label-efficiency curve**: PCK with 1%, 10%, 100% of CUB + L | CUB test, H | Does self-supervised pre-training beat ImageNet init at equal labels? This is the paper's core claim, checked on our data. |
| Eye PCK and visibility calibration (reliability curve, ECE) | H with owner-checked points | Does it work in our shooting conditions, and are confidences meaningful? |
| Coverage: frames with a verified eye | H (#377 bird frames), fed cascade boxes | Does top-down plus cascade close the 147/238 gap? |
| Within-burst eye-sharpness AUC (downstream) | #415 labelled bursts, when available | Does better eye placement improve culling? |
| Local comparison only: the paper's released CUB regressor | CUB test, H | Where do we stand against the published model? Its outputs are measurements only, never labels. |

**Promotion gate:**
- PCK on H ≥ `eye-pose-v0` with a CI-excluded difference.
- Coverage on H ≥ 90% of owner-labelled bird frames with a cascade box.
- Visibility ECE ≤ 0.05.
- If the self-supervised backbone doesn't beat ImageNet init at 10% labels, drop stages 1–2 and keep
  the top-down design alone.

## Milestones

| # | Deliverable | Exit |
|---|---|---|
| S0 | Crop corpus builder: bird boxes → padded crops, burst dedupe, folder splits, H exclusion, manifest | corpus counts and a size histogram |
| S1 | Invariant pre-training (MoCo-style) | kNN retrieval sanity check: same-species crops retrieve each other above an ImageNet backbone |
| S2 | Equivariant projector | landmark matching across warped pairs (PCK of transferred points on CUB) |
| S3 | Top-down keypoint head, fine-tuned on C (+L) | CUB PCK and label-efficiency curve |
| S4 | Domain evaluation on H plus calibration | gate above |
| S5 | ONNX export plus contract v0.2 fields; backend provider (shadow, #426) | backend reads points for the primary region |

## Risks

- **Burst near-duplicates** collapse contrastive learning. Mitigate with per-burst caps and
  folder-level sampling.
- **Small subjects** (< 2% of the frame) give crops with few eye pixels. Cut crops from the full
  rendition, not the thumbnail, and record a `small_subject` limitation.
- **Compute on 8 GB:** start with ResNet-18, mixed precision and a small queue.
- **Label quality:** agent-proposed points must be verified, and owner checks decide the gate.
- **Licence drift:** no code, weights or outputs from the authors' repository enter training without
  permission. CUB-derived weights keep their research-licence caveat.

## Related

- [eye-evidence-spec.md](../architecture/eye-evidence-spec.md): MVP-2 goals (relative eye sharpness, facing, calibration)
- [BIRD_DETECTION.md](../architecture/BIRD_DETECTION.md): detector gap and training priorities
- `training/teacher/`: the RTMDet teacher, whose boxes also feed set U
- Backend: [#408](https://github.com/synthet/image-scoring-backend/issues/408) cascade, [#426](https://github.com/synthet/image-scoring-backend/issues/426) keypoint provider, [#415](https://github.com/synthet/image-scoring-backend/issues/415) labelled bursts
