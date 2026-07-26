# Skill compilation (specialized harnesses)

Pattern adapted from [Compiling an AI agent skill](https://vivekhaldar.com/articles/compiling-an-ai-agent-skill/)
(Vivek Haldar / Token Shrinker method). Sibling implementation with real harnesses:
[`image-scoring-backend/.agent/SKILL_COMPILATION.md`](https://github.com/synthet/image-scoring-backend/blob/main/.agent/SKILL_COMPILATION.md).

## Why

Natural-language `SKILL.md` files are excellent for **discovering** a workflow. Once the same
procedure runs repeatedly, paying a frontier model to re-plan, rebuild state, and re-interpret fixed
rules is wasteful (the "reasoning tax").

**Compile** the crystallized steps into deterministic code. Keep the model only for semantic
judgment. Keep humans for consequential actions.

## Partition rules

| Owner | Owns |
|-------|------|
| **Code** | Paths, config parsing, run invocation, metric extraction from eval output, checkpoint and manifest bookkeeping, keep/revert mechanics, report skeletons |
| **LLM** | Which change to try next, whether a metric delta is real or noise, AC verdicts from messy evidence, commit wording |
| **Human** | Commit, push, tag, weight upload/publish, dataset changes, consequential overrides |

The boundary that matters most here: **hyperparameter choice is judgment.** A harness that picks the
next configuration by rule is a worse search than the model, and it hides why a change was tried.

## Layout

```text
scripts/agent_skills/<name>.py      # deterministic CLI; stdlib only; --json for agents
.cursor/skills/<name>/SKILL.md      # thin bootloader: when-to-use, invoke, LLM slots
tests/test_agent_skills.py          # fixture tests; must not require a GPU
```

Author under `.cursor/` only; run `python scripts/sync_assistant_trees.py` so `.claude/` mirrors it.

## Compiled in this repo

None yet.

| Skill / command | Harness | LLM slots |
|-----------------|---------|-----------|
| — | — | — |

## How to compile another skill

Run [`/compile-skill <name>`](../.cursor/commands/compile-skill.md), which drives this checklist:

1. Confirm the workflow is stable (same sources, filters, and state every run).
2. Partition steps into code / LLM / human — do not freeze judgment into brittle rules.
3. Implement the harness with a read-only default and `--json`.
4. Shrink `SKILL.md` to a bootloader that points at the harness and lists judgment slots.
5. Add fixture tests under `tests/` that pass without a GPU.
6. Sync trees and update [SKILL_INVENTORY.md](SKILL_INVENTORY.md).

## Measurement note

This repository does not store per-session token telemetry, so we do not claim numeric token savings.
Savings are **structural**: agents load a short bootloader and run code instead of re-deriving the
procedure from long prose. Replay harnesses against fixtures to prove behavioral parity.
