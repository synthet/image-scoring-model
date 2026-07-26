---
type: Documentation Hub
title: AI Workflow and Asset Map
description: Where every agent asset lives and the SDLC loop they support.
resource: ai-workflow/README.md
tags: [docs, agents, workflow]
timestamp: 2026-07-26T00:00:00Z
okf_version: 0.1
---

# AI workflow and asset map

## Where agent assets live

| Asset | Location | Notes |
|-------|----------|-------|
| Cursor rules | `.cursor/rules/*.mdc` | **Canonical** always-on guidance |
| Cursor commands | `.cursor/commands/*.md` | **Canonical** slash commands |
| Cursor skills | `.cursor/skills/*/SKILL.md` | **Canonical** skills |
| Claude mirror | `.claude/{rules,commands,skills}` | **Generated** from `.cursor/` — do not hand-edit |
| Agent governance | `.agent/` | Safety, inventories, project guide |
| Wiki | `docs/` | OKF-aligned LLM wiki |

**Single source of truth:** edit assets under `.cursor/` + `.agent/`, then run
`python scripts/sync_assistant_trees.py` to regenerate `.claude/`.

## The SDLC loop

```
/spec → /plan → /decompose → /implement → /test-and-fix → /pr-ready
```

| Phase | Artifact | Gate |
|-------|----------|------|
| `/spec` | Spec with EARS `AC-n` criteria | User approves; no criterion AMBIGUOUS |
| `/plan` | Implementation plan (files, tests, rollback) | User approves |
| `/decompose` | Parallelizable subtasks + reducer (when needed) | Coverage of AC |
| `/implement` | Minimal-diff change set | Lint + narrowest tests green |
| `/test-and-fix` | Root-cause fixes for failures | Targeted suite green |
| `/pr-ready` | DoD report + paste-ready PR text | Ready for human review |

Wiki maintenance: `/wiki-ingest`, `/wiki-query`, `/wiki-lint` (read [WIKI_SCHEMA.md](../WIKI_SCHEMA.md) first).

## Verification

```bash
python scripts/sync_assistant_trees.py --check
python scripts/ci/check_agent_frontmatter.py
python scripts/generate_agent_asset_inventory.py --check
python scripts/okf_lint.py --exclude-prefix archive/
python scripts/wiki_lint.py --exclude-prefix archive/
```

See also [AGENTS.md](../../AGENTS.md), [PROJECT_GUIDE.md](../../.agent/PROJECT_GUIDE.md), and [CANONICAL_SOURCES.md](../CANONICAL_SOURCES.md).
