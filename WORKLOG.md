# Work Log

## 2026-06-23 — Finish pose fine-tune, promote weights, validate Stage-1 localization

### Context
Reviewed two Cursor session exports against the real repo state. The chat logs ended mid-training,
but `results.csv` showed the CUB-200 pose fine-tune had reached epoch 79/100 with strong, plateaued
metrics. However `models/eye_pose_v0.pt` was still the 5.6 MB 3-epoch **smoke** checkpoint (June 14) —
`train_pose.py` only copies `best.pt` on natural completion and every prior run was killed, so the
production weights were never updated. The pipeline was running near-useless localization weights.

Decision (user): finish the fine-tune to epoch 100 (auto-promotes), then validate.

### Environment note
ML deps (torch 2.7.1+cu118 CUDA, ultralytics 8.4.67, editable `eye_quality`) live only in the
**Microsoft Store Python 3.11**: `C:\Users\dmnsy\AppData\Local\Microsoft\WindowsApps\python.exe`.
The shell's default `python` is the agent venv and has no torch.

### Step 1 — Finish fine-tune to epoch 100 (auto-promote) ✅
- Resumed: `train_pose.py --resume runs/pose/runs/pose/wildlife_bird-2/weights/last.pt --epochs 100
  --device 0 --workers 2 --cache disk --output models/eye_pose_v0.pt`.
- Ran epochs 80→100 (21 epochs) in 0.72 h on the RTX 4060, no paging/CUDA errors.
- Final val metrics: pose mAP50 ≈ 0.994, **pose mAP50-95 ≈ 0.983**; box mAP50 ≈ 0.994, box mAP50-95 ≈ 0.885.
- `best.pt` (optimizer stripped → ~5.7 MB) copied to `models/eye_pose_v0.pt`. Verified by fresh
  timestamp (2026-06-23 20:32), not size — the 16 MB file was the with-optimizer training checkpoint.

### Step 1b — Harden promotion ✅
- Added `--promote-only <ckpt>` flag to `training/train_pose.py`: copies a checkpoint to `--output`
  and exits, so an interrupted run can publish weights without a clean training exit. Syntax-verified.

### Step 2 — Validate Stage-1 localization ✅
- `pytest -m gpu` → 1 passed (GPU inference test against promoted weights).
- New `training/eval_localization.py` (reuses `PoseLocalizer` + `load_oriented_image`) over the
  1752-image val split → saved `runs/eval_localization_val.json`:
  - detection rate **99.77%**, visible-eye rate **99.71%**, no-eye **0%**
  - mean eye confidence **0.876**; visibility: 1827 visible / 213 turned_away / 33 partially_visible
- CLI end-to-end (`eye_quality score --debug-dir`) produces eye-centered debug crops at 0.999
  detection confidence. Crops on dark-feathered albatrosses look dark/small (CUB images are low-res),
  but placement is confirmed by keypoint confidence and the eval coverage.

### Finding (not fixed — motivates MVP 2)
The MVP-1 **heuristic** focus fusion under-scores sharp eyes: a Laysan Albatross with
`sharpness_score` 1.0 fused to `focus_score` 0.36 (grade 1) via the clarity/edge/size factors in
`scoring/aggregate.py:fuse_focus_score`. Scoring-tuning issue, not localization.

### Artifacts changed/added
- `training/train_pose.py` — `--promote-only` flag
- `training/eval_localization.py` — new localization coverage QA script
- `models/eye_pose_v0.pt` — promoted epoch-100 weights
- `runs/eval_localization_val.json` — eval summary

### Next (Step 3 roadmap — not started)
1. MVP 2 — learned eye-quality classifier (replaces heuristic fusion; needs labeled eye crops).
2. Expose `subject_bbox_norm` + BioCLIP species crop handoff.
3. Backend + gallery integration (phase 2, per `docs/BACKEND_INTEGRATION.md`).
4. MVP 3 — stack-aware ranker.
