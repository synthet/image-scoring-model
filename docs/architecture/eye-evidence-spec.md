---
type: Architecture Plan
title: Eye evidence spec (MVP-2 direction)
description: Clean-room functional spec for eye keypoints, eye visibility/facing and burst-relative eye sharpness; fixes the MVP-1 under-scoring of sharp eyes and defines the payload the backend localization rollout consumes.
resource: architecture/eye-evidence-spec.md
tags: [docs, architecture, eye-quality, keypoints, scoring, clean-room]
timestamp: 2026-09-24T00:00:00Z
okf_version: 0.1
status: proposed
---

# Eye evidence spec (MVP-2 direction)

> **Status:** proposal. It complements [PIPELINE.md](PIPELINE.md) and the stable
> [API contract](../technical/API_CONTRACT.md). New fields are **additive**.

## Provenance (clean-room)

This spec comes from a competitive analysis of a commercial wildlife burst-culling application's
documented and observable behaviour. That application uses a dedicated bird head/eye keypoint model
and an eye-sharpness criterion.

This page describes **what such a component must do**. It contains no code, identifiers, weights or
fitted constants from that product, and it does not name its models. All numbers are starting points
to re-fit on our own labels.

## Where this fits

The backend's [early localization rollout](https://github.com/synthet/image-scoring-backend/blob/master/docs/architecture/pipeline/localization-rollout.md)
creates versioned subject regions. The backend's
[subject-aware culling evidence](https://github.com/synthet/image-scoring-backend/blob/master/docs/planning/subject-aware-culling-evidence.md)
plan defines six subject-conditioned criteria, and this package owns **criterion 2 (eye)**:

- **Stage 4:** this package runs as a *keypoint provider* on the primary bird region, and its points
  are persisted as region-linked artifacts.
- **Stage 6:** the eye criterion (sub-score, band, limitations) feeds the code-owned burst ranker.

## Functional requirements

### 1. Keypoints

- **Input:** a crop around one bird region, padded by a named crop policy, in display orientation.
- **Output:** per point (`left_eye`, `right_eye`, ideally `bill_tip`, `crown`), each with:
  - normalized x/y in the **region's source rendition**
  - confidence
  - `visible` / `occluded` / `not_present`
- **Head box** is derived from the points and the region, and used for patch sizing.
- **Must not hallucinate:** a turned-away head should give low confidence. Visibility calibration on
  held-out data is an acceptance metric, alongside mAP.
- **Targeted second pass:** if the estimated head is smaller than ~40 px in the first-pass crop,
  re-crop from a larger rendition (the backend's "fine" rendition) and re-run. Record which pass
  produced the points.

### 2. Facing and visibility

| State | Condition | Starting eye-criterion effect |
|---|---|---|
| `both_eyes` | two confident eyes (frontal) | full weight |
| `profile` | one confident eye, the other `not_present` | full weight (the normal bird portrait) |
| `three_quarter_away` | one low-confidence eye | reduced, e.g. ×0.7 |
| `away` / `head_unverified` | no confident eye on an animal | eye sub-score floor; the backend ranker applies its own penalty |

Facing direction (bill vector) is also exported. The backend's composition criterion uses it for
look-room.

### 3. Eye sharpness, relative not absolute

MVP-1 fuses absolute sharpness, clarity and edge energy, then multiplies by size and noise factors.
The worklog's Laysan albatross case shows the failure: sharpness 1.0, but the fused focus came out at
0.36. White, low-contrast plumage lowers *clarity* and *edge energy*, and those are properties of the
subject, not of focus.

Required behaviour:

- Measure on the **fine greyscale rendition**, with a patch centred on the eye whose side is
  proportional to the **head size** (starting at ~0.35 × head width, minimum 16 px). Don't use a
  fixed pixel crop.
- Report `eye_sharpness_rel = sharpness(eye patch) / sharpness(head or subject region)` from the same
  frame. A sharp eye on a sharp head is ~1. Back-focus on the body gives ≪ 1.
- Keep **evidence confidence separate from quality**:
  - tiny eyes, noise and low contrast lower `eye_evidence_confidence`
  - they do **not** multiply the sharpness score down
  - this follows the backend epic invariant that confidences are never multiplied into one score
- Use noise-corrected sharpness (subtract the noise contribution, in the style of Immerkaer). Don't
  apply a separate `noise_penalty` multiplier.
- **Burst-relative output:** the backend ranks eyes within a burst. This package emits raw and
  relative values plus a band. Grade thresholds (`score_to_grade`) stay as a display convenience,
  not as the ranking signal.

### 4. Bands and failure types

Bands: `eye_sharp`, `eye_slightly_soft`, `eye_soft`, `eye_not_visible`, `head_unverified`.

Existing `FailureType` values (defocus, motion blur, turned away, too small, noise, overexposed) map
onto these bands plus `limitations`: `small_eye`, `second_pass`, `low_contrast_plumage`,
`catchlight_clipped`.

## Additive contract fields (proposed `eye_quality` v0.2)

| Field | Type | Meaning |
|---|---|---|
| `facing` | enum | `both_eyes` \| `profile` \| `three_quarter_away` \| `away` |
| `head_bbox_norm` | float[4] | derived head box |
| `eye_sharpness_rel` | float | eye/head sharpness ratio |
| `eye_evidence_confidence` | float 0–1 | reliability of the eye measurement (size, noise, contrast) |
| `band` | enum | see §4 |
| `limitations` | string[] | see §4 |
| `keypoint_pass` | `first` \| `second` | which pass produced the points |
| `keypoints` | list | named points with confidence and visibility |

Existing fields keep their meaning. `focus_score` becomes a monotonic function of
`eye_sharpness_rel` and absolute sharpness, re-fitted on labels.

## Evaluation

- **Labels:**
  - backend human `pick_status` within bursts (primary)
  - the skills repo's `bird-crop-label` best/good/reject sets (secondary; agent labels, not human truth)
- **Metrics:**
  - within-burst AUC of `eye_sharpness_rel` vs MVP-1 `focus_score`
  - visibility calibration (reliability curve)
  - small-head slice
  - white/black plumage slice (albatross, egret, crow, cormorant)
- **Acceptance:**
  - beats MVP-1 on within-burst AUC, folder-grouped
  - no regression on the 1752-image localization split

## Roadmap

1. Relative sharpness + separate evidence confidence, heuristic only. S.
2. Head-proportional patch + second pass. S–M.
3. Facing/visibility states + contract v0.2. S.
4. Learned eye-quality head (the original MVP-2), trained on burst-relative pairs, not absolute
   grades. M.
5. Stack-aware ranker (MVP-3), handled by the backend Arm B ranker; this package supplies features
   only.

## Licence notes

Train only on CUB-200-2011 (research licence; check before redistributing weights), our own labels
and other open part-annotated sets. Do not distil from, or use outputs of, any proprietary eye model.
