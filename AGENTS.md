# AGENTS.md — eye-quality

Agent contract for this repository. Read this before making changes.

**What this is:** wildlife eye localization (YOLO pose) and heuristic eye-focus scoring for
stack-relative culling. Python package `eye_quality` under `src/`, CLI entry point `eye-quality`.

## Commands

```bash
pip install -e ".[dev]"           # install with dev extras

python -m pytest                  # full suite
python -m pytest -m "not gpu"     # fast subset — no weights or CUDA needed
python -m pytest -m gpu           # requires models/eye_pose_v0.pt + CUDA

eye-quality score path/to/bird.jpg --debug-dir /tmp/eye-debug
eye-quality batch manifest.jsonl --output results.jsonl
```

Pretrained weights come from Hugging Face and are **not** in git:

```bash
huggingface-cli download synthet/eye-pose-v0 eye_pose_v0.pt --local-dir models/
```

`models/eye_pose_v0.pt` is used automatically when present.

On Windows PowerShell, put multi-line Python in a `.py` file rather than `python -c "..."`. Nested
quotes get mangled by PowerShell before Python sees them.

## Test vocabulary

Two kinds only — say which one you ran:

| Name | Command | Needs |
|------|---------|-------|
| **Fast subset** | `python -m pytest -m "not gpu"` | nothing beyond dev extras |
| **GPU tests** | `python -m pytest -m gpu` | CUDA + `models/eye_pose_v0.pt` |

A green fast subset proves nothing about detection or scoring on real weights. Report the
distinction rather than saying "tests pass".

## Documentation

Authority map: [docs/CANONICAL_SOURCES.md](docs/CANONICAL_SOURCES.md). Wiki hub: [docs/INDEX.md](docs/INDEX.md).
Folder map: [LAYOUT.md](LAYOUT.md).

| Guide | Description |
|-------|-------------|
| [LAYOUT.md](LAYOUT.md) | Top-level folder semantics |
| [docs/README.md](docs/README.md) | Documentation hub |
| [docs/architecture/PIPELINE.md](docs/architecture/PIPELINE.md) | Architecture and scoring |
| [docs/architecture/BIRD_DETECTION.md](docs/architecture/BIRD_DETECTION.md) | Bird bbox detection and BioCLIP species crops |
| [docs/guides/TRAINING.md](docs/guides/TRAINING.md) | CUB-200 bootstrap and fine-tuning |
| [docs/technical/API_CONTRACT.md](docs/technical/API_CONTRACT.md) | JSON output schema — a contract, not an implementation detail |
| [docs/guides/BACKEND_INTEGRATION.md](docs/guides/BACKEND_INTEGRATION.md) | Backend and gallery wiring |

Downstream consumers (`image-scoring-backend`, `image-scoring-gallery`) depend on the output schema.
Changing a field name or score range is a breaking change — treat
`docs/technical/API_CONTRACT.md` as the source of truth and update it in the same change.

## Agent assets

`.cursor/` is **canonical**; `.claude/` is generated. Never hand-edit `.claude/`.

```bash
python scripts/sync_assistant_trees.py          # regenerate .claude/ from .cursor/
python scripts/sync_assistant_trees.py --check  # fail if out of sync
python scripts/ci/check_agent_frontmatter.py    # frontmatter contract
python scripts/generate_agent_asset_inventory.py --check
python scripts/okf_lint.py --exclude-prefix archive/
python scripts/wiki_lint.py --exclude-prefix archive/
```

| Asset | Location |
|-------|----------|
| Rules (always on) | `.cursor/rules/*.mdc` |
| Slash commands | `.cursor/commands/*.md` |
| Skills | `.cursor/skills/*/SKILL.md` |
| Governance | [`.agent/PROJECT_GUIDE.md`](.agent/PROJECT_GUIDE.md), [`.agent/AGENT_INFRA_INVENTORY.md`](.agent/AGENT_INFRA_INVENTORY.md), [`.agent/SKILL_INVENTORY.md`](.agent/SKILL_INVENTORY.md), [`.agent/SAFETY.md`](.agent/SAFETY.md), [`.agent/SKILL_COMPILATION.md`](.agent/SKILL_COMPILATION.md) |
| Wiki / AI workflow | [docs/ai-workflow/README.md](docs/ai-workflow/README.md) |

**Loop:** `/spec → /plan → /implement → /test-and-fix → /pr-ready`.
Before any unattended training run or sweep, write the contract from
[`autonomous-run-contract`](.cursor/skills/autonomous-run-contract/SKILL.md).

## Safety

[`.agent/SAFETY.md`](.agent/SAFETY.md) governs. The rules that bite most often here:

- Weights, checkpoints, and datasets are never committed.
- Publishing weights, tagging a release, or changing the output contract are human decisions.
- Do not change the evaluation code or validation split in the same change that claims a metric win.
- Source images are read-only; debug output goes to `--debug-dir`.
