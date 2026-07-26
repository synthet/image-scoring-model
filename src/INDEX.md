# `src/` — package source

Installable Python package **`eye_quality`** (see `pyproject.toml` / entry point `eye-quality`).

| Path | Semantics |
|------|-----------|
| `eye_quality/cli.py` | CLI: `score`, `batch`, `detect` |
| `eye_quality/pipeline.py` | End-to-end scoring orchestration |
| `eye_quality/schemas.py` | Pydantic / API-shaped result models |
| `eye_quality/localization/` | Pose + bird detect wrappers |
| `eye_quality/crop/` | Native-resolution eye crops |
| `eye_quality/scoring/` | Heuristics and aggregation |

Contract for JSON output: [`docs/technical/API_CONTRACT.md`](../docs/technical/API_CONTRACT.md).  
Pipeline design: [`docs/architecture/PIPELINE.md`](../docs/architecture/PIPELINE.md).

Do not put datasets, weights, or Ultralytics run logs here.
