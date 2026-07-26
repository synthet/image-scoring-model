---
type: Documentation Schema
title: Wiki Schema
description: Documentation structure, naming, link, metadata, and maintenance conventions.
resource: WIKI_SCHEMA.md
tags: [docs, schema, okf, maintenance]
timestamp: 2026-07-26T00:00:00Z
okf_version: 0.1
---

# Wiki schema — `docs/`

This repository keeps `docs/` as an LLM-maintained wiki: small pages, clear hubs, and stable links to canonical contracts.

## OKF alignment

`docs/` is maintained as an [Open Knowledge Format adoption bundle](OKF_ADOPTION.md): markdown files with YAML frontmatter, stable relative paths as concept identities, relative markdown links as the knowledge graph, folder `INDEX.md` hubs, and append-only `log.md` history.

New living pages and materially edited living pages should include YAML frontmatter with at least `type`; recommended fields are `title`, `description`, `resource`, `tags`, `timestamp`, and `okf_version`. See [OKF_ADOPTION.md](OKF_ADOPTION.md).

## Page types and folders

| Folder | Purpose |
|--------|---------|
| [`architecture/`](architecture/) | System overview and design docs |
| [`guides/`](guides/) | Operator how-tos (training, integration) |
| [`technical/`](technical/) | Stable reference: API contract and deep-dives |
| [`ai-workflow/`](ai-workflow/) | Agent asset map and SDLC loop |
| [`reference/`](reference/) | Generated artifacts (e.g. agent-asset inventory) |
| `planning/`, `features/`, `reports/`, `archive/` | Optional; create with an `INDEX.md` when first used |

### Repo root hub pages (`docs/*.md`)

Thin entry points and **redirect stubs** for previously flat paths (`API_CONTRACT.md`, `PIPELINE.md`, …). Deep content lives in the taxonomy folders.

## Naming

- **New pages:** prefer `kebab-case.md` in new folders.
- **Legacy technical reference:** keep existing `UPPER_CASE.md` names under `architecture/`, `guides/`, and `technical/` to avoid churn.
- **Reports:** include a date when the note is a snapshot (`topic-YYYY-MM-DD.md`).

## Links

- Use **relative** links from the page you are editing.
- Prefer linking contracts via [`CANONICAL_SOURCES.md`](CANONICAL_SOURCES.md) from agent-oriented prose.
- **Cross-repo:** use full GitHub URLs when the canonical doc lives in backend or gallery.

## Indexes and activity log

After adding, renaming, or removing pages:

1. Update the nearest folder `INDEX.md` and, when relevant, [`INDEX.md`](INDEX.md) and [`README.md`](README.md).
2. Append a line to [`log.md`](log.md) under the current month heading:  
   `- YYYY-MM-DD: <verb> — <details and paths>`  
   Verbs: `ingested`, `created`, `updated`, `lint-fixed`, `filed-back`, `reorganized`.

## Agent asset source of truth

Author rules, commands, and skills under **`.cursor/`**. Regenerate `.claude/` with `python scripts/sync_assistant_trees.py`. Do not hand-edit `.claude/`.

## Slash commands

Project commands `/wiki-ingest`, `/wiki-query`, `/wiki-lint` under `.cursor/commands/` (mirrored to `.claude/`) should read this file before large wiki edits.
