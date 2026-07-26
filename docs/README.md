---
type: Documentation Hub
title: Documentation
description: Human entry hub for the eye-quality documentation wiki.
resource: README.md
tags: [docs, hub]
timestamp: 2026-07-26T00:00:00Z
okf_version: 0.1
---

# Documentation

Wildlife eye localization and focus scoring for stack-relative culling.

Full navigation: **[INDEX.md](INDEX.md)**. Authority map: **[CANONICAL_SOURCES.md](CANONICAL_SOURCES.md)**.

## Guides

| Document | Description |
|----------|-------------|
| [architecture/PIPELINE.md](architecture/PIPELINE.md) | End-to-end flow, keypoint schema, scoring heuristics |
| [architecture/BIRD_DETECTION.md](architecture/BIRD_DETECTION.md) | Bird bounding boxes, BioCLIP species crops, detect-only training |
| [guides/TRAINING.md](guides/TRAINING.md) | CUB-200 bootstrap, fine-tuning, resume, validation |
| [technical/API_CONTRACT.md](technical/API_CONTRACT.md) | JSON output schema and field semantics |
| [guides/BACKEND_INTEGRATION.md](guides/BACKEND_INTEGRATION.md) | Wiring into image-scoring-backend and gallery (phase 2) |

## Quick start

```bash
pip install -e ".[dev]"
python -m eye_quality score path/to/bird.jpg --debug-dir /tmp/eye-debug
```

## Pretrained weights

Download fine-tuned pose weights from Hugging Face: **[synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0/tree/main)**

```bash
huggingface-cli download synthet/eye-pose-v0 eye_pose_v0.pt --local-dir models/
```

When `models/eye_pose_v0.pt` exists, the CLI uses it automatically. Otherwise it falls back to `models/yolo11n-pose.pt` (not suitable for wildlife).

## Repository layout

```
src/eye_quality/       # Scoring package (localization, crop, heuristics, CLI)
training/              # Dataset conversion and YOLO fine-tune scripts
data/                  # Local datasets (gitignored)
models/                # Checkpoints + base YOLO weights (gitignored; see models/README.md)
runs/                  # Ultralytics training logs (gitignored)
tests/                 # Unit and optional GPU inference tests
docs/                  # LLM wiki (OKF-aligned); see INDEX.md
hf_upload/             # Hugging Face model card (pose)
hf_upload_detect/      # Hugging Face model card (detect)
scripts/               # Agent-tree sync, wiki lint, CI checkers
.agent/                # Safety, skill inventory, project guide
.cursor/               # Canonical agent rules, commands, skills
.claude/               # Generated mirror of .cursor/ (do not hand-edit)
```

## Tests

```bash
python -m pytest -m "not gpu"
python -m pytest -m gpu   # requires models/eye_pose_v0.pt + CUDA
```
