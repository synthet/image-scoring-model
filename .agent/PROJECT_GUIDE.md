# Project guide — how agents navigate eye-quality

A fast orientation for AI agents. Read this, then the canonical sources it points to.

## 1. Authority (don't invent contracts)

- [`../docs/CANONICAL_SOURCES.md`](../docs/CANONICAL_SOURCES.md) — where each API/schema/command/safety rule is defined.
- [`SAFETY.md`](SAFETY.md) — secrets, weights, datasets, evaluation integrity, contracts.
- [`../AGENTS.md`](../AGENTS.md) — build/test commands and test vocabulary.

## 2. Core workflows

| Goal | Use |
|------|-----|
| Spec a change | `/spec` |
| Plan it | `/plan` |
| Break into parallel work | `/decompose` |
| Implement | `/implement` |
| Fix failing tests | `/test-and-fix` |
| Prepare a PR | `/pr-ready` |
| Docs | `/wiki-ingest`, `/wiki-lint`, `/wiki-query` |
| Unattended training / sweep | skill `autonomous-run-contract` |

## 3. Environment and commands

- Build/test/lint: [`../AGENTS.md`](../AGENTS.md)
- Verified one-liners: [`INFRA_QUICKSTART.md`](INFRA_QUICKSTART.md)

## 4. Asset map

- Full inventory: [`AGENT_INFRA_INVENTORY.md`](AGENT_INFRA_INVENTORY.md)
- Where every asset lives + the SDLC loop: [`../docs/ai-workflow/README.md`](../docs/ai-workflow/README.md)
- Manual skill review table: [`SKILL_INVENTORY.md`](SKILL_INVENTORY.md)
