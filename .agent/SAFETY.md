# Agent safety and hygiene — eye-quality

## Secrets and credentials

- Never commit `.env`, API keys, or Hugging Face tokens. `HF_TOKEN` lives in the environment, not in
  a file in this repo.
- Never paste live credentials into prompts, logs, or tool arguments.

## Weights and large artifacts

- **Do not commit model weights.** `models/*.pt` comes from Hugging Face
  ([`synthet/eye-pose-v0`](https://huggingface.co/synthet/eye-pose-v0)) or from a training run; it is
  fetched, not versioned.
- Do not commit checkpoints, TensorBoard runs, debug overlays, or datasets. Keep them gitignored.
- **Never overwrite a released checkpoint** in place. New runs write to a run-scoped directory.

## Publishing

- Uploading weights to Hugging Face, tagging a release, or changing what
  [`docs/technical/API_CONTRACT.md`](../docs/technical/API_CONTRACT.md) promises are **human decisions**. An agent
  prepares them and stops.

## Datasets

- Local training data lives under `data/` (gitignored). Do not rewrite, re-split, or prune it as a
  side effect of another task.
- `datasets/` holds local smoke fixtures (gitignored). Do not commit them.
- Changing the validation split invalidates every previously recorded metric. If it must change, say
  so explicitly and re-baseline — do not compare across a split change.

## Evaluation integrity

- Do not modify the evaluation code or metric definition in the same change that claims a metric
  improvement.
- Report the fast-subset caveat: `python -m pytest -m "not gpu"` proves nothing about GPU paths.

## Contracts

- Do not invent CLI flags, JSON output fields, or score ranges. The output schema is
  [`docs/technical/API_CONTRACT.md`](../docs/technical/API_CONTRACT.md); downstream consumers are the backend and gallery
  ([`docs/guides/BACKEND_INTEGRATION.md`](../docs/guides/BACKEND_INTEGRATION.md)).
  Authority map: [`docs/CANONICAL_SOURCES.md`](../docs/CANONICAL_SOURCES.md).

## Git

- Never modify `.git/config` or add non-standard git extensions.
- Commit and push only when the user asks.

## Source images

- Images passed to `eye-quality score` are user photographs. Read them; do not move, rename, or
  modify them. Debug output goes to `--debug-dir`.
