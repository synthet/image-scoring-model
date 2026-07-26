# `.cursor/` — canonical agent assets

**Source of truth** for Cursor/Claude assistant trees in this repo.

| Path | Semantics |
|------|-----------|
| `rules/*.mdc` | Always-on guidance |
| `commands/*.md` | Slash commands (`/spec`, `/wiki-lint`, …) |
| `skills/*/SKILL.md` | Reusable skills |

After edits, regenerate the Claude mirror:

```bash
python scripts/sync_assistant_trees.py
python scripts/sync_assistant_trees.py --check
```

Do not hand-edit [`.claude/`](../.claude/INDEX.md). Inventory: [`docs/reference/agent-asset-inventory.md`](../docs/reference/agent-asset-inventory.md).
