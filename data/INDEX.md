# `data/` — local training datasets

**Gitignored.** Large CUB-derived YOLO datasets produced by `training/` converters.

Typical layout:

| Path | Semantics |
|------|-----------|
| `wildlife_bird/` | Pose dataset (boxes + 6 keypoints) |
| `wildlife_bird_det/` | Detect-only dataset (boxes) |
| `eye_dataset/` / `eye_dataset_val/` | Optional eye-crop / eval sets |

Do not rewrite or re-split as a side effect of another task; changing the val split invalidates metrics ([`.agent/SAFETY.md`](../.agent/SAFETY.md)).

Small smoke fixtures belong under [`datasets/`](../datasets/INDEX.md), not here.
