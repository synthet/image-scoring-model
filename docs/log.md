# Wiki activity log

Append-only. One line per wiki restructure, newest under the current month.
Format: `- YYYY-MM-DD: <verb> — <details and paths>` (verbs: `created`, `updated`, `ingested`,
`lint-fixed`, `reorganized`).

## 2026-09

- 2026-09-25: created — [reports/teacher-pseudo-labels-v0-2026-09-25.md](reports/teacher-pseudo-labels-v0-2026-09-25.md) (new [reports/INDEX.md](reports/INDEX.md)) and [guides/TEACHER_PSEUDO_LABELS.md](guides/TEACHER_PSEUDO_LABELS.md): upstream RTMDet-tiny teacher (own PyTorch implementation, 2638/2638 parity), stratified pool with #377 folder exclusion, calibrated label policy, 877 pseudo-labelled images; section added to [architecture/BIRD_DETECTION.md](architecture/BIRD_DETECTION.md); `training/teacher/` added to `training/INDEX.md`.
- 2026-09-25: created — [planning/eye-location-density.md](planning/eye-location-density.md): eye location as a 2D density (expected eye sharpness, overlap with the focus map, uncertainty), options heatmap / Gaussian / SimCC marginals, deciding experiment.
- 2026-09-25: created — [planning/ssl-landmark-pretraining.md](planning/ssl-landmark-pretraining.md): self-supervised landmark pre-training plan (method reimplemented from the ICCV 2021 paper; authors' unlicensed code and weights are evaluation-only), top-down keypoints fed by the backend cascade, milestones S0–S5 and gate; indexed in [planning/INDEX.md](planning/INDEX.md).

- 2026-09-25: created — [planning/reference-workflow-improvement-candidates.md](planning/reference-workflow-improvement-candidates.md): clean-room proposal for explicit evidence semantics, targeted detail inference, candidate preservation, uncertainty, project-owned calibration, runtime conformance, active learning, and diagnostics; added [planning/INDEX.md](planning/INDEX.md) and linked both documentation hubs.
- 2026-09-24: updated — [architecture/BIRD_DETECTION.md](architecture/BIRD_DETECTION.md): measured gap vs an open COCO detector on the backend #377 cohort (82% recall at 4% vs 63% FP) and training priorities (hard negatives, small subjects, raptors).
- 2026-09-24: created — [architecture/eye-evidence-spec.md](architecture/eye-evidence-spec.md): clean-room MVP-2 spec (targeted second-pass keypoints, facing states, head-relative eye sharpness, separate evidence confidence, additive contract v0.2) aligned with the backend localization rollout; linked from [architecture/INDEX.md](architecture/INDEX.md).

## 2026-07

- 2026-07-26: created — root `LAYOUT.md` plus per-folder `INDEX.md` files describing folder semantics (`src/`, `training/`, `tests/`, `scripts/`, `data/`, `datasets/`, `runs/`, `.agent/`, `.cursor/`, `.claude/`).
- 2026-07-26: reorganized — moved HF staging from `hf_upload/` and `hf_upload_detect/` to `hf/eye-pose-v0/` and `hf/bird-detect-v0/`.
- 2026-07-26: reorganized — adopted OKF wiki taxonomy; moved PIPELINE/BIRD_DETECTION → architecture/, TRAINING/BACKEND_INTEGRATION → guides/, API_CONTRACT → technical/; added CANONICAL_SOURCES, WIKI_SCHEMA, OKF_ADOPTION, INDEX hubs, redirect stubs, ai-workflow, reference inventory.
