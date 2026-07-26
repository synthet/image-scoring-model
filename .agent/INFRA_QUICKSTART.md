# Infra quickstart — verified one-liners

Run from repo root. See [`AGENTS.md`](../AGENTS.md) for the full agent contract.

## Install and tests

```bash
pip install -e ".[dev]"
python -m pytest -m "not gpu"     # fast subset
python -m pytest -m gpu           # needs CUDA + models/eye_pose_v0.pt
```

## Agent trees

```bash
python scripts/sync_assistant_trees.py
python scripts/sync_assistant_trees.py --check
python scripts/ci/check_agent_frontmatter.py
python scripts/generate_agent_asset_inventory.py
python scripts/generate_agent_asset_inventory.py --check
```

## Wiki lint

```bash
python scripts/okf_lint.py --profile project --exclude-prefix archive/
python scripts/wiki_lint.py --exclude-prefix archive/
```

## CLI smoke

```bash
eye-quality score path/to/bird.jpg --debug-dir /tmp/eye-debug
```

## Weights

```bash
huggingface-cli download synthet/eye-pose-v0 eye_pose_v0.pt --local-dir models/
huggingface-cli download synthet/bird-detect-v0 bird_detect_v0.pt --local-dir models/
```

Base Ultralytics weights live under `models/` (`yolo11n.pt`, `yolo11n-pose.pt`, …) — gitignored.
