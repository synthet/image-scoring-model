# `scripts/` — repo tooling

Maintainability helpers for agents and CI — not the scoring runtime.

| Script | Semantics |
|--------|-----------|
| `sync_assistant_trees.py` | Regenerate `.claude/` from canonical `.cursor/` (`--check`) |
| `ci/check_agent_frontmatter.py` | Skill/command/rule frontmatter contract |
| `okf_lint.py` / `okf_bundle.py` | OKF frontmatter lint on `docs/` |
| `wiki_lint.py` / `wiki_lint_scan.py` | Broken links, orphans, INDEX coverage |
| `generate_agent_asset_inventory.py` | Write `docs/reference/agent-asset-inventory.md` from `.cursor/` |

See [`.agent/INFRA_QUICKSTART.md`](../.agent/INFRA_QUICKSTART.md) for verified one-liners.
