---
description: Always-on safety and secret-handling rules.
alwaysApply: true
---

# Safety & secrets (always on)

- Secrets (API keys, tokens, `HF_TOKEN`) live in `.env` or the environment (git-ignored). Never commit
  them, never paste them into prompts/logs/tool args, never return them in tool output.
- Never modify `.git/config` or add non-standard git extensions (no `extensions.worktreeConfig`).
- Treat write-capable / destructive tools (training runs, checkpoint writes, dataset edits, execute
  code) as high-risk: prefer read-only diagnostics unless the user explicitly asks for the write.
- Never commit model weights, checkpoints, or datasets; never overwrite a released checkpoint.
- Publishing weights, tagging a release, or changing `docs/API_CONTRACT.md` are human decisions.
- Validate external inputs; side-effecting actions need confirmation/approval.

Full detail: [`.agent/SAFETY.md`](../../.agent/SAFETY.md).
