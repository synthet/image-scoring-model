# Agent infrastructure inventory — eye-quality

Catalog of agent-facing paths in this repository. Companion to [`SKILL_INVENTORY.md`](SKILL_INVENTORY.md)
and [`docs/ai-workflow/README.md`](../docs/ai-workflow/README.md).

**Canonical tree:** `.cursor/` — regenerate `.claude/` with `python scripts/sync_assistant_trees.py`.
Never hand-edit `.claude/`.

**Last reviewed:** 2026-07-26.

## Trees

| Path | Purpose | Status |
|------|---------|--------|
| `.cursor/rules/*.mdc` | Always-on rules (canonical) | active |
| `.cursor/commands/*.md` | Slash commands (canonical) | active |
| `.cursor/skills/*/SKILL.md` | Skills (canonical) | active |
| `.claude/rules/*.md` | Generated mirror of mirrored rules | generated |
| `.claude/commands/*.md` | Generated mirror of commands | generated |
| `.claude/skills/` | Generated mirror of skills | generated |
| `.agent/SAFETY.md` | Safety SoT | active |
| `.agent/SKILL_INVENTORY.md` | Manual skill/command review table | active |
| `.agent/SKILL_COMPILATION.md` | Compiled-harness notes | active |
| `.agent/PROJECT_GUIDE.md` | Agent navigation router | active |
| `.agent/INFRA_QUICKSTART.md` | Verified one-liners | active |
| `.agent/AGENT_INFRA_INVENTORY.md` | This catalog | active |
| `.agent/scratch/` | Local experiments (gitignored) | local-only |

## Wiki and docs

| Path | Purpose | Status |
|------|---------|--------|
| `docs/` | OKF LLM wiki | active |
| `docs/CANONICAL_SOURCES.md` | Authority map | active |
| `docs/WIKI_SCHEMA.md` / `OKF_ADOPTION.md` | Wiki conventions | active |
| `docs/technical/API_CONTRACT.md` | JSON output contract | active |
| `docs/reference/agent-asset-inventory.md` | Generated from `.cursor/` | generated |

## Scripts

| Path | Purpose |
|------|---------|
| `scripts/sync_assistant_trees.py` | `.cursor/` → `.claude/` sync (`--check`) |
| `scripts/ci/check_agent_frontmatter.py` | Skill/command/rule frontmatter CI |
| `scripts/okf_lint.py` | OKF frontmatter lint on `docs/` |
| `scripts/wiki_lint.py` | Broken links / orphans / INDEX coverage |
| `scripts/generate_agent_asset_inventory.py` | Regen agent-asset inventory (`--check`) |

## Mirrored rules (`.cursor` → `.claude`)

Only these rule stems are synced (see `MIRROR_RULES` in `scripts/sync_assistant_trees.py`):

- `karpathy-coding`
- `safety-and-secrets`
- `sdlc-core`

## Drift watchlist

- After any skill/command/rule change: run sync and inventory `--check` in the same PR.
- After wiki moves: update stubs, `INDEX.md` hubs, `log.md`, and `CANONICAL_SOURCES.md`.
- Do not flip SoT to `.claude/` without an explicit migration plan.
