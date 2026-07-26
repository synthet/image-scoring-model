# Repository layout

Semantics of top-level folders in **image-scoring-model** (`eye-quality`). For the documentation wiki, start at [`docs/INDEX.md`](docs/INDEX.md). For agent navigation, see [`.agent/PROJECT_GUIDE.md`](.agent/PROJECT_GUIDE.md).

| Folder | Role | Tracked? | Index |
|--------|------|----------|-------|
| [`src/`](src/INDEX.md) | Installable `eye_quality` package (CLI, pipeline, scoring) | yes | [INDEX](src/INDEX.md) |
| [`training/`](training/INDEX.md) | Dataset conversion, YOLO train/eval scripts and configs | yes | [INDEX](training/INDEX.md) |
| [`tests/`](tests/INDEX.md) | Pytest suite (fast subset + GPU-marked) | yes | [INDEX](tests/INDEX.md) |
| [`docs/`](docs/INDEX.md) | OKF LLM wiki (contracts, guides, architecture) | yes | [INDEX](docs/INDEX.md) |
| [`models/`](models/README.md) | Local checkpoints + base YOLO weights | README only; `*.pt` ignored | [README](models/README.md) |
| [`hf/`](hf/README.md) | Hugging Face model-card staging (no weights) | yes | [README](hf/README.md) |
| [`scripts/`](scripts/INDEX.md) | Agent-tree sync, wiki lint, CI helpers | yes | [INDEX](scripts/INDEX.md) |
| [`data/`](data/INDEX.md) | Large local training/eval datasets | gitignored | [INDEX](data/INDEX.md) |
| [`datasets/`](datasets/INDEX.md) | Small local smoke fixtures (e.g. coco8) | gitignored | [INDEX](datasets/INDEX.md) |
| [`runs/`](runs/INDEX.md) | Ultralytics training/eval artifacts | gitignored | [INDEX](runs/INDEX.md) |
| [`.agent/`](.agent/INDEX.md) | Safety, inventories, project guide, scratch | mostly yes; `scratch/` ignored | [INDEX](.agent/INDEX.md) |
| [`.cursor/`](.cursor/INDEX.md) | Canonical agent rules, commands, skills | yes | [INDEX](.cursor/INDEX.md) |
| [`.claude/`](.claude/INDEX.md) | Generated mirror of `.cursor/` — do not hand-edit | yes (generated) | [INDEX](.claude/INDEX.md) |

Ephemeral / local-only (no durable contract): `debug_crops/` (CLI debug output, gitignored), `.pytest_cache/`, `.venv/`.

## Authority

| Question | Source |
|----------|--------|
| What may agents change? | [`AGENTS.md`](AGENTS.md), [`.agent/SAFETY.md`](.agent/SAFETY.md) |
| Where is each contract defined? | [`docs/CANONICAL_SOURCES.md`](docs/CANONICAL_SOURCES.md) |
| How is the wiki maintained? | [`docs/WIKI_SCHEMA.md`](docs/WIKI_SCHEMA.md) |
