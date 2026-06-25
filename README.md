# eye-quality

Wildlife eye localization (YOLO pose) and heuristic eye-focus scoring for stack-relative culling.

## Install

```bash
pip install -e ".[dev]"
```

GPU training requires CUDA-enabled PyTorch. See [docs/TRAINING.md](docs/TRAINING.md).

## Quick start

```bash
python -m eye_quality score path/to/bird.jpg --debug-dir /tmp/eye-debug
```

When `models/eye_pose_v0.pt` exists, it is used automatically.

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Architecture and scoring |
| [docs/BIRD_DETECTION.md](docs/BIRD_DETECTION.md) | Bird bbox detection and BioCLIP species crops |
| [docs/TRAINING.md](docs/TRAINING.md) | CUB-200 bootstrap and fine-tuning |
| [docs/API_CONTRACT.md](docs/API_CONTRACT.md) | JSON output schema |
| [docs/BACKEND_INTEGRATION.md](docs/BACKEND_INTEGRATION.md) | Backend and gallery wiring |

## CLI

```bash
eye-quality score path/to/image.jpg
eye-quality batch manifest.jsonl --output results.jsonl
```

## Tests

```bash
python -m pytest -m "not gpu"
python -m pytest -m gpu   # requires models/eye_pose_v0.pt + CUDA
```
