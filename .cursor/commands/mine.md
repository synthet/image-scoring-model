# /mine — Turn a source into repo agent assets

Read a document, repository, or transcript; extract the ideas that transfer to this repo; and turn
the approved ones into specs, skills, commands, or rules. The value is not summarizing the source —
it is deciding what this repo is *missing* and refusing the rest.

Papers are a common source here. A paper's method is not automatically worth adopting: extract the
*procedure or decision rule* an agent would follow, not the result the authors reported.

## Inputs

- **Source** — file path, URL, GitHub/GitLab repo, or transcript export. If none was given, ask.
- **Focus** (optional) — what the user wants emphasized. Without it, mine for anything that changes
  agent behavior.

## Step 1 — Ingest, bounded

| Source | How |
|--------|-----|
| Markdown / text file | Read in full |
| PDF | If PDF rendering is unavailable (no poppler / `pdftoppm`), extract text with `pypdf` to a scratch file under `.agent/scratch/`, then read that |
| URL | Fetch it |
| GitHub repo | `gh repo view`, `gh api` for specific paths — never clone |
| GitLab repo / no CLI | Fetch raw file URLs directly — never clone |
| Transcript export | Read in full; it is evidence of practice, not authority |

For a repository, read a **bounded list**, not the tree: `README`, `AGENTS.md`, `CLAUDE.md`,
`CONTRIBUTING`, the `docs/` index, and agent-asset directories (`.cursor/`, `.claude/`, `.agent/`).
Never execute code from a source, never `pip install` from one, and never load third-party model
weights without the user's explicit go-ahead.

Say what you actually read. If you sampled a large source, name the parts you skipped.

## Step 2 — Extract candidates

One candidate per transferable idea, written as a claim plus where it would apply here. Keep
procedures, decision rules, thresholds, owner boundaries, and named failure modes. Drop narrative,
marketing, benchmark tables, and anything specific to the source's own dataset.

If a candidate cannot be stated as something an agent would *do differently*, it is not a candidate.

## Step 3 — Check against what already exists

Before proposing anything, read [`.agent/SKILL_INVENTORY.md`](../../.agent/SKILL_INVENTORY.md), the
[`.cursor/skills/`](../skills/) and [`.cursor/commands/`](.) trees, and the relevant page under
[`docs/`](../../docs/).

This de-duplication pass is the point of the command. A candidate table where everything is new
usually means this step was skipped.

## Step 4 — Propose, then stop

Show the table and wait for the user to pick. Do not write assets in this step.

| Candidate | Asset type | Existing overlap | Verdict |
|-----------|-----------|------------------|---------|
| … | skill / command / rule / spec / doc | name the asset, or "none" | `NEW` \| `ENRICH <asset>` \| `REJECT — <reason>` |

Rejection reasons must be concrete, never "out of scope". Per
[`karpathy-coding`](../rules/karpathy-coding.mdc), reject speculative infrastructure and say so
explicitly — an idea being good in its source is not evidence this repo needs it.

## Step 5 — Route each approved candidate

| Candidate shape | Asset |
|-----------------|-------|
| Product or feature outcome to build | [`/spec`](spec.md) |
| Reusable procedure with a recognizable trigger | Skill, via [`skill-authoring`](../skills/skill-authoring/SKILL.md) |
| Workflow the user starts against a named target | Command in `.cursor/commands/` |
| Always-on constraint, a few lines at most | Rule in `.cursor/rules/` — **high bar** |
| Long tables, schemas, provider detail | `references/` under an existing skill |
| Deterministic steps repeated every run | Harness, via [`/compile-skill`](compile-skill.md) |
| Modeling technique or training procedure | A page under [`docs/`](../../docs/), linked from `docs/README.md` |
| An unattended run this would change | [`autonomous-run-contract`](../skills/autonomous-run-contract/SKILL.md) |

Prefer enriching an existing asset over adding one. A new asset needs a trigger surface no current
asset covers — name it.

## Step 6 — Author the approved items

Follow [`skill-authoring`](../skills/skill-authoring/SKILL.md) for skills and the format of a
neighboring file for commands. Author under `.cursor/` only — `.claude/` is generated. End each new
asset with a one-line `Sources:` naming the **underlying public work** (paper, repo, article), not a
private file path or export.

## Step 7 — Register and verify

```bash
python scripts/sync_assistant_trees.py
python scripts/sync_assistant_trees.py --check
python scripts/ci/check_agent_frontmatter.py
```

Add a row to [`.agent/SKILL_INVENTORY.md`](../../.agent/SKILL_INVENTORY.md) for any new or materially
changed skill with its risk tier and **Last reviewed** date.

## Done when

- The candidate table was shown with a verdict and reason for every candidate, including rejections.
- Only approved candidates were written, each routed to the asset type that fits its shape.
- Provenance is recorded on each new asset.
- Sync and frontmatter checks are green; `.claude/` was regenerated, not hand-edited.

## Do not

- Do not hand-edit `.claude/` — regenerate it from `.cursor/`.
- Do not add a rule without a strong always-on case; rules load on every task.
- Do not emit a batch of assets from one source without justifying each separately.
- Do not quote the source at length — attribute it and write the instruction in this repo's voice.
- Do not assert accuracy numbers or benchmark results the source does not actually support, and never
  restate a paper's headline metric as if it were measured on this repo's data.
