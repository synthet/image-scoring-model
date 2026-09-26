---
type: Plan
title: Evidence model improvement candidates
description: Clean-room proposals for calibrated, auditable, subject-aware evidence models and runtime contracts.
resource: docs/planning/reference-workflow-improvement-candidates.md
tags: [planning, models, evidence, calibration, clean-room]
timestamp: 2026-09-25T18:00:00Z
okf_version: 0.1
status: proposed
---

# Evidence model improvement candidates

> Provenance: derived from competitive analysis of a commercial application's observable behaviour and documentation; contains no code, identifiers, fitted constants or model artefacts from it.

This page proposes behavior-level improvements for the project-owned model stack. It is a research agenda, not a commitment or implementation specification. Candidate models, training data, calibration curves, thresholds, and runtime settings must come from open sources or project-owned measurements.

Related pages:

- [Eye evidence specification](../architecture/eye-evidence-spec.md)
- [Subject detection](../architecture/BIRD_DETECTION.md)
- [Training guide](../guides/TRAINING.md)
- [API contract](../technical/API_CONTRACT.md)

## 1. Publish evidence, confidence, and applicability separately

Each prediction should distinguish three questions:

- **Evidence value:** what was measured, such as subject extent, head pose, visible-eye focus, motion, or occlusion.
- **Confidence:** how reliable that measurement appears under the current image conditions.
- **Applicability:** whether the measurement is meaningful for this image at all.

The output should preserve `unknown`, `not applicable`, and `failed` as distinct states. A missing or unusable eye region must not silently become a low-quality eye score. This separation lets downstream ranking degrade gracefully and explain why evidence was omitted.

## 2. Use a coarse-to-detail measurement path

Run inexpensive whole-frame localization first, then spend high-resolution inference only on plausible subjects and detail regions. The second pass should be scheduled by expected decision value: small subjects, close ranking margins, uncertain landmarks, and images likely to become a burst winner deserve more detail than obvious rejects.

Measure accuracy and latency together. The useful comparison is not merely which model is most accurate, but which routing policy improves within-burst ordering under a fixed compute budget.

## 3. Preserve multiple subject candidates

Return a bounded candidate set when an image contains several plausible subjects. For every candidate, expose the evidence used to select the primary subject and retain alternatives for downstream review. Selection should combine salience, size, visibility, centrality, and temporal consistency without assuming that the largest subject is always the intended one.

Manual subject selection should remain authoritative, and later model runs should not overwrite it silently.

## 4. Cross-check regions, landmarks, and silhouettes

Independent signals should validate one another. A head landmark outside the selected subject, an eye landmark outside the head region, or a silhouette that disagrees strongly with the detector should lower confidence or trigger abstention. These checks should emit machine-readable reason codes so consumers can distinguish occlusion, truncation, low resolution, and model disagreement.

## 5. Model visibility and facing explicitly

Represent visible, partly visible, occluded, out-of-frame, and indeterminate detail states. For bilateral features, report which side is visible rather than fabricating a paired measurement. Facing estimates should support profile and rear-facing cases and should abstain when the available pixels do not support a stable conclusion.

## 6. Add relative detail evidence

Absolute sharpness varies with sensor, lens, processing, crop size, and noise. In addition to raw measurements, evaluate relative detail quality among comparable frames from the same capture sequence. Compare equivalent subject regions at matched scale and record whether a result came from absolute or relative evidence.

The model package should provide measurements and uncertainty. Final keep/reject policy belongs to the ranking layer so users can change preferences without rerunning inference.

## 7. Calibrate on project-owned labels

Build a labeled set that includes small subjects, heavy crops, high noise, motion blur, occlusion, multiple subjects, backlighting, and out-of-domain species. Calibrate confidence and quality estimates against these labels, with held-out splits that prevent adjacent frames from leaking across train and validation sets.

Inspect calibration by subject size, camera conditions, and failure category. Any numeric hyperparameter or threshold proposed during exploration is only a local starting point and must be re-fitted on project-owned data before production use.

## 8. Support open-set behavior

Taxonomy predictions should expose uncertainty and an explicit unknown outcome. Temporal propagation may help within a coherent sequence, but it should preserve provenance and must not override manual labels. Evaluate whether confidence remains meaningful for taxa and scenes absent from training.

## 9. Turn reviewer disputes into an active-learning queue

Capture cases where users change the selected subject, disagree with an eye or head location, or reverse a ranking. Sample from these disputes alongside hard negatives and low-confidence cases. Deduplicate near-identical frames and maintain capture-group splits so evaluation reflects new sessions rather than memorized sequences.

## 10. Evaluate candidate models under their real contract

For each open candidate, document licensing, input assumptions, preprocessing, supported platforms, output semantics, and known failure modes. Compare candidates on:

- subject localization recall and false positives;
- landmark or region localization error relative to subject size;
- evidence availability and abstention quality;
- confidence calibration;
- within-sequence ordering and winner recall;
- latency, peak memory, and export fidelity;
- robustness to crop scale, orientation, metadata loss, and color conversion.

Aggregate quality scores should not hide weak subgroups. Publish cohort-level results and representative failure categories.

## 11. Require export and runtime conformance

Treat the training framework and deployed runtime as separate implementations that must agree. Maintain a small, redistribution-safe conformance set and compare output shapes, coordinate transforms, missing-value behavior, tolerances, determinism, and error handling after every export change.

The runtime should report model identity, contract version, preprocessing version, device, and execution provider with each result. This makes regressions diagnosable without coupling consumers to training internals.

## 12. Produce inspectable diagnostics

When requested, emit compact overlays and structured evidence for the selected subject, alternative candidates, regions, landmarks, confidence, applicability, and failure reasons. Diagnostics must be optional, redactable, and safe to omit in normal operation. They should help answer whether a poor decision came from localization, measurement, calibration, or downstream ranking.

## Suggested delivery order

1. Stabilize the additive evidence contract and missing-state semantics.
2. Establish project-owned labels, sequence-safe splits, and runtime conformance fixtures.
3. Add candidate preservation and region/landmark consistency checks.
4. Introduce the targeted detail pass behind a measured compute budget.
5. Calibrate uncertainty and relative detail evidence.
6. Add active-learning and open-set evaluation after the basic contract is stable.

## Acceptance evidence

A candidate is ready for downstream shadow use when:

- output semantics and versioning are documented;
- unknown, inapplicable, and failed results remain distinguishable;
- training and deployed runtimes pass the conformance set;
- calibration and ordering metrics improve on held-out capture groups;
- degraded cases abstain predictably instead of emitting confident placeholders;
- licensing and model provenance are recorded; and
- diagnostics identify the failing stage without exposing private source material.

## Non-goals

- Reproducing a third party's model weights, learned tables, or internal architecture.
- Encoding final culling policy inside the measurement model.
- Treating one aggregate benchmark as sufficient for production promotion.
- Replacing manual labels or subject choices without an explicit user action.

