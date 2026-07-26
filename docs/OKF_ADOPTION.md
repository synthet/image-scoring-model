---
type: Documentation Governance
title: Open Knowledge Format Adoption
description: Local OKF profile for docs/ — frontmatter, indexes, log, and lint.
resource: OKF_ADOPTION.md
tags: [docs, okf, agents, governance]
timestamp: 2026-07-26T00:00:00Z
okf_version: 0.1
---

# Open Knowledge Format adoption

This repository treats `docs/` as an **OKF-aligned knowledge bundle**: markdown concept files with YAML frontmatter, relative cross-links, folder indexes, and an append-only activity log.

## Why this structure

| OKF idea | Local convention |
|---|---|
| Bundle | `docs/` |
| Concept document | Any non-archive markdown page that describes one topic or contract |
| Concept identity | Stable relative file path under `docs/` |
| Frontmatter | YAML block with at least `type`; recommended fields below |
| Links as graph | Relative markdown links between docs and code |
| Index / log | Uppercase `INDEX.md` hubs and append-only `log.md` |

## Frontmatter profile

```yaml
---
type: Technical Reference
title: Human-readable page title
description: One sentence explaining the page's purpose.
resource: technical/EXAMPLE.md
tags: [docs]
timestamp: 2026-07-26T00:00:00Z
okf_version: 0.1
---
```

### Required field

- `type`: document category. Consumers must tolerate unknown values.

### Recommended fields

- `title`, `description`, `resource`, `tags`, `timestamp`, `okf_version` (`0.1`)

## Type vocabulary

| Type | Use for |
|---|---|
| `Documentation Hub` | Entry points such as `README.md` or redirect stubs |
| `Documentation Index` | `INDEX.md` navigation pages |
| `Documentation Schema` | Wiki structure and maintenance rules |
| `Documentation Governance` | OKF / process docs |
| `Source-of-Truth Map` | Authority maps (`CANONICAL_SOURCES.md`) |
| `Technical Reference` | API, pipeline, detection, implementation references |
| `Guide` | Training and integration walkthroughs |
| `Report` | Point-in-time audits |
| `Archive` | Historical pages retained for traceability |

## Folder and index rules

1. Keep the taxonomy from [WIKI_SCHEMA.md](WIKI_SCHEMA.md).
2. Update the nearest `INDEX.md` when adding, removing, or moving a page.
3. Update root [INDEX.md](INDEX.md) / [README.md](README.md) for new hubs or contracts.
4. Append to [log.md](log.md) for every wiki restructure.
5. Flat-path stubs at `docs/*.md` redirect to taxonomy homes; do not remove them without updating external links.

## Automated lint

```bash
python scripts/okf_lint.py --profile project --exclude-prefix archive/
python scripts/wiki_lint.py --exclude-prefix archive/
python scripts/okf_lint.py --json --fail-on error --exclude-prefix archive/
```

Redirect stubs (`type: Documentation Hub` with `tags` including `redirect`) are valid living pages.

## Agent workflow

1. Read [CANONICAL_SOURCES.md](CANONICAL_SOURCES.md) before changing contract or schema claims.
2. Use this OKF profile for metadata.
3. Prefer many small concept pages over duplicated mega-docs.
4. Keep [log.md](log.md) append-only.

## Official OKF reference

- [OKF SPEC v0.1](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
- [OKF README](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/README.md)

This repo does **not** depend on Google's enrichment-agent package.
