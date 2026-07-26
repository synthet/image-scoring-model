# /compile-skill — Lower a stable skill into a deterministic harness

Use when a procedural skill has crystallized and agents keep paying the reasoning tax to re-derive
the same steps. Compiles the fixed parts into `scripts/agent_skills/<name>.py` and shrinks the skill
to a thin bootloader. Pattern and partition rules:
[`.agent/SKILL_COMPILATION.md`](../../.agent/SKILL_COMPILATION.md).

## Inputs

- Target skill: `.cursor/skills/<name>/SKILL.md` (or a command). If the user did not name one, list
  uncompiled candidates with the stability evidence you have and ask — do not pick silently.
- Stability evidence: repeated transcripts running the same procedure, or a row in
  [`.agent/SKILL_INVENTORY.md`](../../.agent/SKILL_INVENTORY.md) with real usage.

## Readiness gate

Compile only if **all** hold. If any fails, say "not ready" and name the failing condition instead of
compiling an unstable procedure.

- Same sources, filters, and state every run — no per-run re-planning.
- At least one step is pure mechanics (path resolution, config parsing, metric extraction from eval
  output, checkpoint bookkeeping, report skeletons).
- The judgment left over is nameable in a sentence or two ("choose the next hyperparameter to try").
- The skill is not mostly judgment. `systematic-debugging` and `karpathy-guidelines` have nothing to
  lower — leave them as prose.

## Step 1 — Partition the steps

Produce this table before writing code, and show it to the user:

| Owner | Gets |
|-------|------|
| **Code** | Paths, config parsing, run invocation, metric extraction, checkpoint and manifest bookkeeping, keep/revert mechanics, report skeletons |
| **LLM** | Which change to try next, whether a metric delta is real or noise, AC verdicts from messy evidence, commit wording |
| **Human** | Commit, push, tag, weight upload/publish, dataset changes, consequential overrides |

If a step needs a heuristic to stay correct, it belongs to the LLM. Freezing judgment into rules is
the main way this pattern fails — a harness that picks hyperparameters by rule is a worse search than
the model.

## Step 2 — Implement the harness

```text
scripts/agent_skills/<name>.py
```

- Stdlib only — no new runtime dependency for a helper script. Resolve the repo root relative to the
  script file; no hardcoded absolute paths.
- Read-only by default. Inspect/plan run free; anything that writes (config edits, checkpoint
  pruning) needs an explicit `apply` subcommand or `--run`.
- Support `--json` for agents and a readable summary otherwise. Errors to stderr, non-zero exit.
- Emit a `needs_llm_judgment` marker rather than guessing when evidence is ambiguous.
- The harness never commits, pushes, tags, uploads weights, or writes to `datasets/`.

## Step 3 — Shrink the skill to a bootloader

Keep `name` first and a non-empty `description` in frontmatter, then keep only: when to use,
**Invoke** (copy-pasteable commands), **LLM judgment slots** (numbered), **Human authority**, and
**Verify**. Delete prose the harness now enforces.

## Step 4 — Test

Add fixture tests under `tests/` and run:

```bash
python -m pytest tests/test_agent_skills.py -q
```

Assert behavioral parity with the prose procedure on at least one realistic fixture — that is the
evidence the compile was lossless. Do not require a GPU for harness tests.

## Step 5 — Sync and record

```bash
python scripts/sync_assistant_trees.py
python scripts/sync_assistant_trees.py --check
python scripts/ci/check_agent_frontmatter.py
python -m pytest -m "not gpu"
```

Then update [`.agent/SKILL_COMPILATION.md`](../../.agent/SKILL_COMPILATION.md) (new row in the
compiled table) and [`.agent/SKILL_INVENTORY.md`](../../.agent/SKILL_INVENTORY.md) (note the harness
path, refresh **Last reviewed**).

## Done when

- Partition table was shown and the LLM slots are named in the bootloader.
- Harness runs read-only by default, emits `--json`, and hands ambiguity back to the model.
- Fixture tests pass without a GPU and demonstrate parity on a real case.
- Sync and frontmatter checks are green; `.cursor/` is canonical and `.claude/` was regenerated.

## Do not

- Do not hand-edit `.claude/` — it is generated from `.cursor/` by `sync_assistant_trees.py`.
- Do not give the harness commit/push/tag, weight-publish, or dataset-write authority.
- Do not compile hyperparameter choice into a rule.
- Do not claim numeric token savings; this repo has no per-session telemetry. Savings are structural.
