# Documentation

Wildlife eye localization and focus scoring for stack-relative culling.

## Guides

| Document | Description |
|----------|-------------|
| [PIPELINE.md](PIPELINE.md) | End-to-end flow, keypoint schema, scoring heuristics |
| [BIRD_DETECTION.md](BIRD_DETECTION.md) | Bird bounding boxes, BioCLIP species crops, detect-only training |
| [TRAINING.md](TRAINING.md) | CUB-200 bootstrap, fine-tuning, resume, validation |
| [API_CONTRACT.md](API_CONTRACT.md) | JSON output schema and field semantics |
| [BACKEND_INTEGRATION.md](BACKEND_INTEGRATION.md) | Wiring into image-scoring-backend and gallery (phase 2) |

## Quick start

```bash
pip install -e ".[dev]"
python -m eye_quality score path/to/bird.jpg --debug-dir /tmp/eye-debug
```

When `models/eye_pose_v0.pt` exists, the CLI uses it automatically. Otherwise it falls back to generic `yolo11n-pose.pt` (not suitable for wildlife).

## Repository layout

```
src/eye_quality/       # Scoring package (localization, crop, heuristics, CLI)
training/              # Dataset conversion and YOLO fine-tune scripts
data/wildlife_bird/    # YOLO pose dataset (gitignored, produced from CUB-200)
models/                # Fine-tuned weights (gitignored)
runs/                  # Ultralytics training logs (gitignored)
tests/                 # Unit and optional GPU inference tests
```

## Tests

```bash
python -m pytest -m "not gpu"
python -m pytest -m gpu   # requires models/eye_pose_v0.pt + CUDA
```
