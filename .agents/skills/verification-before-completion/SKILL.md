---
name: verification-before-completion
description: >-
  Use before claiming work is complete, fixed, passing, ready to commit, or ready
  for PR. Apply to ensure fresh command output supports every success claim and to
  report warnings or failures honestly.
---

# Verification before completion

Do not claim done/fixed/passing/ready until fresh command output supports the claim.

## Preferred proofs (eye-quality)

```bash
python -m pytest                       # full suite
python -m pytest -m "not gpu"          # fast subset; GPU tests need weights
python -m pytest tests/test_foo.py -q  # narrowest scope after a targeted fix
eye-quality score <image>              # CLI still works end to end
python scripts/sync_assistant_trees.py --check
python scripts/ci/check_agent_frontmatter.py
git status --short
```

A run that skipped every `gpu`-marked test has not proven GPU behavior — say so rather than
reporting a clean pass.

## LLM judgment slots

1. **Name the claim** and pick the falsifying proof.
2. **Interpret output** after the final edit — exit code, warnings, skipped tests, scope.
3. Report pass / warn / fail with the exact command; never upgrade incomplete verification to pass.

## Use with

- `validate-implementation` when an AC matrix is required
- [`autonomous-run-contract`](../autonomous-run-contract/SKILL.md) — the same honesty rule applied to a whole unattended training run
