# `tests/` — pytest suite

| Marker / kind | Command | Needs |
|---------------|---------|-------|
| Fast subset | `python -m pytest -m "not gpu"` | Dev extras only |
| GPU tests | `python -m pytest -m gpu` | CUDA + `models/eye_pose_v0.pt` |

A green fast subset does **not** prove detection or scoring on real weights. Vocabulary lives in [`AGENTS.md`](../AGENTS.md).

Keep fixtures small; do not check in CUB-scale data or checkpoints.
