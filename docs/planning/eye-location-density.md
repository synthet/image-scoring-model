---
type: Plan
title: Eye location as a density, not a point
description: Design note on representing the eye as a 2D probability density so eye-sharpness can be computed as an expectation over the focus map; options (heatmap, 2D Gaussian, SimCC marginals), costs and the experiment that decides it.
resource: docs/planning/eye-location-density.md
tags: [planning, keypoints, eye-quality, uncertainty, focus]
timestamp: 2026-09-25T00:00:00Z
okf_version: 0.1
status: proposed
---

# Eye location as a density, not a point

> **Status:** design note, not scheduled. It feeds the MVP-2 evidence in
> [eye-evidence-spec.md](../architecture/eye-evidence-spec.md) and the fine-tuning head in
> [ssl-landmark-pretraining.md](ssl-landmark-pretraining.md).

## The question

Today an eye is a point plus a confidence. Eye sharpness is then measured on a fixed square patch around
that point. Should the model output a **probability density** over where the eye is, so that the question
"is the eye in focus?" becomes an overlap between the eye density and a focus map?

## Short answer

Yes, where the eye is small or uncertain, which is where culling decisions are hardest. At high
resolution it makes little difference.

| Situation | Point + patch | Density |
|---|---|---|
| Large, clear eye | patch sits on the eye; fine | same answer |
| Eye a few pixels wide, point off by 2–3 px | patch half on feathers; sharpness biased low or high | expectation weights only where the eye probably is |
| Occluded, backlit, or eye vs catchlight vs dark feather | one confident-looking point, silently wrong | wide or two-peaked density: uncertainty is visible |
| Head turned, eye barely visible | a point anyway | low total mass → "eye not verified", not "eye soft" |

The density does three jobs that a point can't:
1. **It weights sharpness by location certainty.** Eye sharpness becomes
   `E[s] = Σₓ p(x) · s(x)`, where `s` is a local focus map (e.g. windowed Laplacian energy or a learned
   focus map). Uncertainty of the location enters the score instead of being ignored.
2. **It gives uncertainty for free.** The spread of `p` (or its entropy) is a direct "how sure is the
   location" signal. That separates *eye soft* from *eye not found*, which the evidence spec requires.
3. **It carries a natural overlap measure.** "Is the focus plane on the eye?" is the mass of `p` inside
   the in-focus region: `P(in focus) = Σₓ p(x) · 1[s(x) > τ]`. That is the eye-area/focus-area overlap
   directly.

## Representation options

| Option | What the head outputs | Pros | Cons |
|---|---|---|---|
| **A. 2D heatmap** (top-down, e.g. 64×64 per keypoint) | full density | multi-modal, arbitrary shape; standard in top-down pose | more compute and memory per crop; quantization at small crops |
| **B. 2D Gaussian** (μ, Σ per keypoint) | mean + 2×2 covariance | tiny, differentiable, tilted ellipses; trainable with Gaussian NLL | uni-modal: can't represent "either this spot or that one" |
| **C. SimCC marginals** (1D x and y distributions) | two 1D distributions | already the head type in many light keypoint models; sub-pixel | the product `p(x)·p(y)` assumes independence: an axis-aligned blob, no tilt, and false mass at the corners of two-peaked cases |

**Suggested order:** B first. It adds a few outputs to any head and gives calibrated spread. Move to A only
if the error analysis shows real two-peaked cases, such as eye vs catchlight or two birds' heads in one
crop. Treat C only as a cheap approximation.

## Training and calibration

- Train B with a Gaussian negative log-likelihood, or A with a KL divergence against a target Gaussian
  whose σ scales with eye size. The target σ can come from the annotation spread where several annotators
  agree.
- **Calibration check:** the fraction of true eye points inside the predicted k-σ ellipse should match the
  nominal coverage (39% at 1σ, 86% at 2σ in 2D). Report it on the held-out cohort (set H in the SSL plan).
- Keep the visibility probability separate from the location density. "Is there a visible eye" and
  "where is it" are different questions.

## How the score uses it

```text
eye_sharpness      = Σ p(x) s(x)                    (expected local focus at the eye)
eye_in_focus_prob  = Σ p(x) 1[s(x) > τ_head]        (τ relative to the head/body focus, not absolute)
eye_location_conf  = f(spread of p)                  (→ "eye not verified" when too wide)
```

`τ_head` is relative, as in the evidence spec: the eye is compared with the sharpest part of the head
region, so a uniformly soft frame doesn't pass just because the eye is its sharpest spot.

## The experiment that decides it

On the #415 labelled bursts (and the #377 frames with owner eye checks):
1. Compute eye sharpness three ways: a point + fixed patch (today's approach), a point + a patch scaled
   to eye size, and option B's expectation.
2. Measure the within-burst AUC for keep vs reject, and the rank correlation with owner eye-sharpness
   judgements.
3. **Adopt the density only if** it beats the scaled patch by a confidence-interval-excluded margin, or
   if it cuts the "confident but wrong eye" failures in the error audit.

## Related

- [eye-evidence-spec.md](../architecture/eye-evidence-spec.md): head-relative eye sharpness, facing,
  separate evidence confidence
- [ssl-landmark-pretraining.md](ssl-landmark-pretraining.md): the fine-tuning head this would sit on
- Backend [#426](https://github.com/synthet/image-scoring-backend/issues/426): keypoint provider contract
  (additive fields: `sigma` or `cov` per keypoint)
