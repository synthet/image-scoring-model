---
type: Plan
title: Next actions for bird boxes and eye localization
description: Evaluation, data, and training actions after verifying the owner photo paths.
resource: docs/planning/bird-eye-next-actions-2026-10-05.md
tags: [planning, training, detection, keypoints, evaluation]
timestamp: 2026-10-05T00:00:00Z
okf_version: 0.1
status: active
---

# Bird box and eye model: next actions

Goal: improve field bird detection and visible-eye localization while preserving
the released checkpoints and proving any gain on human ground truth. This is the
execution update to [the retraining plan](bird-eye-retraining-2026-09-27.md).
The [field evaluation contract](../../training/FIELD_EVALUATION.md) defines
coordinates, matching, and metrics.

## Verified starting point

- The owner cohort records WSL paths under `/mnt/d/Photos/`. All **216/216**
  original photos exist after mapping that prefix to `D:/Photos/`. The earlier
  claim that these photos were inaccessible was incorrect.
- The owner presence labels contain 76 `bird`, 139 `no_bird`, and 1 `unsure`.
  Presence labels do **not** establish complete bird boxes or eye centers.
- `data/verified_negatives_v1/manifest.jsonl` is a private, gitignored manifest
  of the 139 owner-reviewed bird-free originals, with source SHA-256 hashes.
  They occupy 113 source folders; none overlaps the independent promotion
  cohort's folders. The manifest is not yet a decoded YOLO training dataset.
- `training/eval_field.py` and `training/run_safety.py` are implemented; focused
  standard-library tests passed (25 tests). Released weights have not been
  replaced, and no candidate has been trained or shown to improve metrics.
- A CUDA-capable Docker image was smoke-checked with PyTorch, Ultralytics, and
  Pillow. A full data decode or training run has not been validated there.

Jev System One (`jev-1.13.0`) selected **freeze human field evaluation** as the
first workstream from these facts (choice confidence 1.0). This is a sequencing
aid, not evidence that labels or model quality are sufficient.

## Ordered actions and gates

| Step | Action | Evidence required to advance |
|---|---|---|
| 1. Freeze the field benchmark | Choose separate calibration and final-test source folders. Decode exact display-oriented renditions. Human-review every bird box, visible eye center, eye state, and genuinely bird-free frame. Record source and rendition hashes; keep bursts and derivatives in one split. Use the existing 207-frame promotion cohort only after deciding and recording whether its prior use disqualifies it as an untouched test. | Versioned private manifest passes `training/eval_field.py` validation; annotation QA confirms complete boxes and eye states; split/group audit reports zero leakage. The original 216-frame owner cohort cannot be promoted to full box/eye GT from presence labels alone. |
| 2. Establish baselines | Export predictions from the exact released bird detector and eye pose weights on identical renditions. Sweep confidence thresholds on calibration, freeze operating points, then score final test once. Include bird instance/frame recall, bird-free FP rate, eye precision, end-to-end visible-eye recall, error by size/condition, and latency. | Report weights, rendition, prediction, evaluator, and manifest hashes; denominators and per-group uncertainty; recorded frozen thresholds. No training change or candidate win is claimed in this step. |
| 3. Prepare detector training data | Decode the 139 verified negatives consistently and verify SHA-256 provenance. Review existing field pseudo-labels; label **all** birds in selected positive frames, especially missed and small birds. Build a new versioned training dataset and keep calibration/test folders out. | Sampled rendition and box QA, explicit positive/negative counts, class balance and source-group audit; empty YOLO labels only for human-reviewed negatives. |
| 4. Run one detector pilot | Write a bounded run contract. Fine-tune from `bird_detect_v1.pt` into a new run directory and `models/candidates/`, first with a short CUDA/data smoke test, then one controlled D1 run. Compare against the frozen baseline at matched image and threshold settings. Test higher resolution only if small-bird errors justify it. | Candidate hash, config and dataset hashes, training log, calibrated recall/FP and latency comparison. Retain only an improvement within the frozen FP budget and without critical slice regression. |
| 5. Prepare eye labels and pilot | Correct false/inferred eye supervision in CUB conversion. Human-label visible eye centers, hidden/unresolvable states, and complete bird boxes on separate field training frames. Train a corrected-CUB P1 candidate first; add field labels in P2. Consider top-down bird crops only if error analysis shows a resolution or upstream-box limit. | Annotation audit; end-to-end visible-eye recall and precision versus the frozen baseline, with upstream bird misses included; crop geometry and latency checks if top-down is attempted. |
| 6. Decide on promotion | Compare candidates on the held-out field test once, document failure slices and independent review, and prepare rollback instructions. | Evidence package supports the preset gates. Publishing weights, changing the backend default, or changing the output contract remains a separate human decision. |

## Immediate next artifact

Create the private, human-annotation packet for Step 1: a source-group inventory,
calibration/test assignment, exact RAW-to-display rendition settings, and
annotation checklist. Do not run D1 or P1 as an improvement claim before that
benchmark and released-weight baseline are frozen. Preparing and auditing the
verified-negative renditions can proceed while annotation is under way, but
the 76 presence-positive frames still require complete boxes.

Before any unattended training, write the
[autonomous run contract](../../.cursor/skills/autonomous-run-contract/SKILL.md)
with epoch and time limits, checkpoint path, stop criteria, and the metric used
to retain or reject a candidate. Keep source photos read-only and datasets,
predictions, reports, and candidate weights under gitignored paths.
