---
type: Plan
title: LLM CLI judges for retraining quality control
description: Reuse the backend's blind vision-judge workflow to audit labels, detect missed birds and eyes, and gate candidate-model evaluation.
resource: docs/planning/llm-cli-retraining-judges.md
tags: [planning, training, evaluation, judges, quality-control]
timestamp: 2026-09-27T00:00:00Z
okf_version: 0.1
status: proposed
---

# LLM CLI judges for retraining quality control

Requested extension to the [recall-first retraining plan](bird-eye-retraining-2026-09-27.md): use vision-capable LLM CLI judges throughout dataset review and candidate evaluation. Missing birds or visible eyes remains the primary concern. Judges produce structured review evidence and can hold a candidate for investigation; numerical benchmarks and human reference labels remain independently reported.

## Existing workflow to reuse

The local backend is `D:\Projects\image-scoring-backend`. Its [blind bbox-panel report](https://github.com/synthet/image-scoring-backend/blob/master/docs/reports/bbox-llm-judge-panel-2026-09-24.md) used Antigravity, Cursor, and a partial Claude Code run to grade 576 unique boxes. Vision-majority identified 106/112 boxes on owner-labelled bird-free frames as WRONG. That validates one presence-related subset; it does not establish 95% accuracy on box geometry, tiny birds, or eyes. Codex lacked quota in that run; availability must be checked anew.

The reusable compiled harness is in sibling `D:\Projects\image-scoring-skills`:

- `.cursor/skills/bird-crop-label/SKILL.md` and `references/workflow.md`.
- `.cursor/scripts/run-bird-crop-label.ps1`: CLI dispatch, chunking, raw outputs, UUID manifest.
- `.cursor/scripts/merge-bird-crop-verdicts.ps1`: deterministic within-burst consensus and CSV update.
- `prompts/burst-judge.md`: existing `best | good | reject` task.

The backend's [labelling runbook](https://github.com/synthet/image-scoring-backend/blob/master/docs/guides/BIRD_CROP_LABELLING.md) defines agent-derived provenance. Its [prior run record](https://github.com/synthet/image-scoring-backend/blob/master/docs/reports/SESSION_BIRD_CROP_CLOSEOUT_2026-08-05.md) records an unreadable-sheet trust failure. These are useful precedents for image-access checks and explicit judge coverage.

**Adapt the dispatch and provenance machinery; add a separate localization-review task.** The current harness is fixed to the old burst roster and writes its `label_set.csv`; it is not a drop-in detector/eye evaluator. Do not overwrite that study or resample its roster. Existing `-WhatIfSetup` is setup-only, but still creates work files; existing merge `-DryRun` creates a proposed CSV. Neither is a new localization-review mode.

Two legacy behaviors must not carry into this task: the burst prompt requests a guessed verdict even if a sheet is unreadable, and its merger forces at least one best frame. Localization judges must be able to abstain, and no evidence must never become an accepted result. The old Claude/Cursor/Codex/Antigravity weights (2/2/1/1) concern burst preference, not validated eye-localization accuracy.

## Review packets and blinding

Prepare immutable packets from identical EXIF-corrected renditions used by the evaluator. Each packet includes opaque item IDs, source/rendition hashes, dimensions, crop transforms, and separate unmarked and marked images. Persist the model-to-arm mapping outside judge access.

| Pass | Judge sees | Required observation |
|---|---|---|
| Presence first | Unmarked full frame plus systematic overlapping tiles, including frames where every model predicts nothing | Bird present / absent / uncertain; possible additional birds; inspectability |
| Bird boxes | Full frame with one anonymous box and its padded crop | TIGHT / LOOSE / PARTIAL / MULTI / WRONG / UNCERTAIN; bird outside box |
| Eye localization | Unmarked head crop plus a separate copy with anonymous eye markers, with full-frame context | Visible / hidden / occluded / unresolved; point on eye / off eye / uncertain; visible eye missed |
| Candidate comparison | Randomized A/B overlays and matched crops; neither arm named | A better / B better / tie / different subject / neither / uncertain, with visible reason |
| Downstream burst audit | Matched-resolution eye/head crops within one burst | Whether a localization change improves usable eye crops or introduces a wrong-subject/focus-ranking regression |

Presence observations are sealed before showing predictions. Use systematic tiles independent of model boxes so the packet can reveal birds missed by both models. A low-resolution overview alone cannot justify a bird-free or no-visible-eye verdict. Supply source-resolution details when available; enlargement cannot recover missing detail. Mark incomplete coverage or inadequate detail as uncertain.

Hide checkpoint names, confidences, pipeline scores, historical labels, and prior judges' votes. Randomize color/order and balance A/B position; save the randomization seed. Mask original filenames and captions that reveal the arm. Treat text visible in photographs as image content, never instructions.

Use the backend's box rubric consistently: a significant clipped body part is PARTIAL; tiny tail/wingtip/feet overhang alone is not; any head or bill cut is PARTIAL. Keep bird count, outside-box birds, and wrong-subject selection separate from box tightness. Eye checks get a new versioned rubric; do not infer eye validity from the old box-panel agreement.

## Judge qualification and consensus

1. Discover installed/authenticated image-capable CLI providers using the existing harness transport. Candidate providers are Claude Code, Codex, Antigravity, and Cursor where supported; record the actual underlying model/version. Two CLI names using the same model are not independent judges.
2. Start with a bounded 24-item development smoke set spanning positives, negatives, small subjects, hidden/visible eyes, and multi-bird scenes. Include an unreadable/missing-image control and a visible nonce or randomized marker that is absent from the text roster. Merely echoing supplied image IDs is not proof of image access.
3. Use two independently qualified vision models per item; invoke a third on disagreements or explicit uncertainty. Keep first-pass judgments isolated. Failed image access, invalid output, unknown IDs, quota errors, or timeout produce an unanswered item, never a favorable verdict.
4. Qualify against human-reviewed development examples, separately for bird presence, box quality, and eyes. Before scaling, freeze minimum task-specific agreement/coverage requirements and inspect all false-absence errors; a provider can qualify for boxes while remaining unqualified for eyes. The image-access controls must all pass.
5. For discrete fields, two matching qualified independent votes form provisional consensus. A three-way split, unresolved count, wrong-subject dispute, or insufficient detail routes to human review. Repeated calls to one provider do not create extra independent votes. In the recall-first policy, any credible minority claim of an overlooked bird/visible eye remains a review item even if two judges say absent.
6. Human-review all unresolved critical misses and a random 10% of agreements, stratified by difficulty and verdict. Audit negatives especially carefully. Measure judge agreement with people, per-class errors, abstention rate, missing coverage, and position bias. Agreement among judges alone is not accuracy.

The smoke budget is 48 initial item judgments plus at most 24 third judgments; this counts item judgments, not necessarily CLI calls. Before each larger batch, record item/provider limits, concurrency, timeout, at most one transient retry, and a token/spend or subscription-quota ceiling in the run contract. Stop dispatch at the cap and preserve partial coverage without silently shrinking the denominator.

## Structured output and provenance

Add a strict localization JSON schema and parser, separate from the old `BURST/END` parser. Proposed fields are internal artifact fields, not additions to the public scoring API:

| Fields | Purpose |
|---|---|
| `run_id`, `item_id`, `arm_id`, `judge_id`, `task`, `rubric_version` | Exact join keys and prompt identity |
| `image_access`, `inspectability`, `status` | Distinguish valid observations, abstention, transport failure, and invalid schema |
| `bird_presence`, `box_grade`, `bird_outside_box` | Presence, box quality, and missed instances |
| `eye_visibility`, `point_on_eye`, `visible_eye_missed`, `wrong_subject` | Eye-specific failure observations; scoped to an identified subject/eye |
| `pairwise_preference`, `visible_reason`, `evidence_panel_ids` | Auditable comparison and its visual basis |

Validate enums, task-specific required fields, exact roster coverage, duplicate IDs, arm membership, and schema version. An absent answer is not `false`. Do not use free-form LLM coordinates as precision ground truth; coordinate error/IoU still comes from verified annotations and deterministic computation.

Keep UUID-stamped packets/manifests, raw responses, parsed verdicts, consensus, adjudication queue, and audit report under an ignored run directory such as `runs/retraining-review/<run-id>/`. Include model checkpoint hashes, dataset/split hashes, image hashes, prompt/rubric hashes, actual CLI/model versions, seeds, call status, and usage. Resume only exact matching item/image/prompt/model identities; never reuse stale verdicts after changing an image or rubric.

All automated results carry `ground_truth_kind: agent-derived`. Store human adjudications separately with reviewer provenance; do not retroactively call the full panel output human ground truth. Keep training, calibration, development review, and final-test outputs in separate manifests.

## Controls on the retraining loop

| Checkpoint | Panel action | Effect |
|---|---|---|
| Before D1 | Review existing field negatives, suspicious pseudo-boxes, and multi-bird omissions | Quarantine disputed examples for verified annotation; block unreviewed negative admission |
| Before P1/P2 | Audit inferred eye targets, hidden-eye cases, and field annotations | Queue label corrections; detect guessed visible-eye supervision |
| After each completed candidate run | Blind v0/candidate comparison on fixed development packets, plus disagreements and a random unchanged sample | HOLD candidates with unresolved new misses, image-access failure, or incomplete required review; prioritize next development work |
| Before promotion | Freeze rubric/judges/thresholds and review the final holdout once alongside deterministic metrics | Require both benchmark gates and resolved critical review items; prepare human promotion decision |

Review every completed experiment's selected checkpoint, rather than dispatching a panel after every epoch. Define mandatory review coverage in advance. Report outcomes as PASS / HOLD / FAIL with enumerated evidence: a HOLD is unresolved review, while FAIL is an established acceptance-gate violation. Judges do not autonomously change thresholds, rewrite labels, adjust evaluation code, or promote weights. Investigate a critical judge flag against the pixels before counting it as a confirmed regression.

LLM outputs identify what to inspect; they are not automatically training labels. Follow the existing teacher guide's restriction on proprietary/unlicensed model outputs as training targets. Corrections require independent human annotation or an approved label source. Test frames never enter the retraining pool. Model selection uses development data; final-test judge feedback stays out of iterative tuning.

Jev may optionally check rubric consistency or classify judgment-free text descriptions, following the backend's aligned-rubric precedent. It is text-only and does not count as another independent vision vote or verify an eye/box by itself. It is not required for the first CLI panel.

## Implementation and checks

Proposed implementation in this repository: `training/review/build_packets.py`, `training/review/judge_schema.py`, `training/review/consensus.py`, versioned rubrics, and an adapter to the sibling harness's CLI transport. Extend the shared transport only if needed; do not modify the legacy burst prompt or merge contract for localization. Backend data access remains read-only, and selected review artifacts remain separate from production scores and `human_labels`.

Add meaningful fast-subset tests in `tests/test_retraining_judges.py`: unreadable image cannot yield a valid vote; quota failure cannot pass a candidate; two calls to one model cannot satisfy quorum; minority missed-bird evidence enters the review queue; unknown/duplicate IDs are rejected; changed packet hashes invalidate cached judgments; incomplete review yields HOLD; test items cannot enter a training correction queue; informed outputs cannot enter blind evaluation labels.

First execution milestone, after implementing the adapter: a 24-item smoke panel on baseline/development images, an image-access/parseability report, and an owner-audit queue. Then scale the qualifying providers to D1/P1 reviews. This planning change ran no live CLI judges and produced no new accuracy measurements.
