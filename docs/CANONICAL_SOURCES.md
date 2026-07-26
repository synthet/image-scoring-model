---
type: Source-of-Truth Map
title: Canonical Sources
description: Authority map — single source of truth for each contract and convention.
resource: CANONICAL_SOURCES.md
tags: [docs, governance, authority]
timestamp: 2026-07-26T00:00:00Z
okf_version: 0.1
---

# Canonical sources

Agents and contributors must check this map before inventing API fields, score ranges, training commands, or agent-asset locations.

| Contract / convention | Source of truth |
|-----------------------|-----------------|
| Public JSON output schema | [`technical/API_CONTRACT.md`](technical/API_CONTRACT.md) |
| Pipeline / scoring design | [`architecture/PIPELINE.md`](architecture/PIPELINE.md) |
| Bird bbox detection | [`architecture/BIRD_DETECTION.md`](architecture/BIRD_DETECTION.md) |
| Training / CUB bootstrap | [`guides/TRAINING.md`](guides/TRAINING.md) |
| Backend / gallery wiring | [`guides/BACKEND_INTEGRATION.md`](guides/BACKEND_INTEGRATION.md) |
| Build / test / lint commands | [`../AGENTS.md`](../AGENTS.md) |
| Safety rules | [`../.agent/SAFETY.md`](../.agent/SAFETY.md) |
| Skill / command inventory (manual) | [`../.agent/SKILL_INVENTORY.md`](../.agent/SKILL_INVENTORY.md) |
| Agent infra path catalog | [`../.agent/AGENT_INFRA_INVENTORY.md`](../.agent/AGENT_INFRA_INVENTORY.md) |
| Agent navigation | [`../.agent/PROJECT_GUIDE.md`](../.agent/PROJECT_GUIDE.md) |
| Agent assets (rules/commands/skills) | [`ai-workflow/README.md`](ai-workflow/README.md) |
| Generated asset catalog | [`reference/agent-asset-inventory.md`](reference/agent-asset-inventory.md) |
| Wiki conventions | [`WIKI_SCHEMA.md`](WIKI_SCHEMA.md) |
| OKF frontmatter profile | [`OKF_ADOPTION.md`](OKF_ADOPTION.md) |
| Weights download | [`../models/README.md`](../models/README.md) |
| HF model-card staging | [`../hf/README.md`](../hf/README.md) |
| Repository folder map | [`../LAYOUT.md`](../LAYOUT.md) |

**Rule:** code and the written contract must never disagree. If you change one, change the other in the same PR. Changing field names or score ranges in the API contract is a **human decision** (see [SAFETY.md](../.agent/SAFETY.md)).

Redirect stubs at `docs/API_CONTRACT.md` (and siblings) exist only for old URLs — the living contract is under `technical/`.
