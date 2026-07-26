# eye-quality

Wildlife eye localization (YOLO pose) and heuristic eye-focus scoring for stack-relative culling.

## Install

```bash
pip install -e ".[dev]"
```

GPU training requires CUDA-enabled PyTorch. See [docs/guides/TRAINING.md](docs/guides/TRAINING.md).

## Pretrained weights

Fine-tuned bird pose weights are hosted on Hugging Face: **[synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0/tree/main)**

```bash
pip install huggingface_hub
huggingface-cli download synthet/eye-pose-v0 eye_pose_v0.pt --local-dir models/
```

When `models/eye_pose_v0.pt` exists, the CLI uses it automatically.

## Quick start

```bash
python -m eye_quality score path/to/bird.jpg --debug-dir /tmp/eye-debug
```

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/README.md](docs/README.md) | Documentation hub |
| [docs/INDEX.md](docs/INDEX.md) | Full wiki index |
| [docs/architecture/PIPELINE.md](docs/architecture/PIPELINE.md) | Architecture and scoring |
| [docs/architecture/BIRD_DETECTION.md](docs/architecture/BIRD_DETECTION.md) | Bird bbox detection and BioCLIP species crops |
| [docs/guides/TRAINING.md](docs/guides/TRAINING.md) | CUB-200 bootstrap and fine-tuning |
| [docs/technical/API_CONTRACT.md](docs/technical/API_CONTRACT.md) | JSON output schema |
| [docs/guides/BACKEND_INTEGRATION.md](docs/guides/BACKEND_INTEGRATION.md) | Backend and gallery wiring |
| [docs/CANONICAL_SOURCES.md](docs/CANONICAL_SOURCES.md) | Authority map |

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
