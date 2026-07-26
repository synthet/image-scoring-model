---
name: autonomous-run-contract
description: >-
  Use before letting an agent run without step-by-step supervision — training
  runs, hyperparameter sweeps, overnight jobs, batch scoring over a dataset,
  "keep iterating until the metric improves", or subagent fan-out. Also use to
  decide how much orchestration a task needs (single run vs ratchet loop vs
  parallel sweep). Produces a written contract with metric, budget, revert rule,
  and stop conditions before the run starts.
---

# Autonomous run contract

Supervised work is governed by the `/spec → /plan → /implement` loop. **Unattended** work is not —
nobody reads the intermediate steps, so the constraints have to be written down before the run
starts. This skill produces that contract.

Training is the loop this pattern was built for: a mutable program, a fixed evaluation, a time
budget, and a git history. The bottleneck is not the next model call — it is whether each experiment
has a parent state, a diff, a metric, and a keep-or-discard decision that survives the session.

## When to use

- "Train it overnight", "sweep the learning rate", "keep tuning until val improves"
- Batch scoring or evaluation across a dataset
- Any loop that mutates training config, weights, or `datasets/` between human checkpoints
- Deciding whether a task needs a sweep at all

## Gate — can success be verified?

If there is no metric that separates better from worse, **stop and define one, or keep the human in
the loop**. For this repo the metric is a validation number from an eval run over a fixed split — not
"the loss looked lower". Autonomy without a verifiable signal produces activity, not progress.

Then confirm the other three loop preconditions:

- **Reversible** — every step can be undone. `git reset` to the last retained commit, and checkpoints
  written to a run-scoped directory so a bad run never overwrites a good one. Weights under `models/`
  that came from Hugging Face are **not** yours to overwrite.
- **Short horizon** — one iteration finishes fast enough to give frequent feedback. Prefer a short
  run on a small split as the inner loop; a full fine-tune is not an inner loop.
- **Bounded environment** — one config surface, one dataset, one metric.

## Step 1 — Pick the smallest architecture that fits

| Situation | Start with | Why |
|-----------|-----------|-----|
| One question about current behavior | Single eval run | Lowest cost; no machinery |
| Metric can be measured after each change | **Ratchet loop** (change → train → eval → keep or revert) | Repeated feedback compounds |
| Independent configs to compare | Parallel sweep | Reduces wall-clock; needs a reducer |
| Alternatives must stay comparable | Branch or tag per retained experiment | Preserves lineage instead of one moving HEAD |
| Results must survive the session | Committed metrics + run manifests | A chat summary is not an experiment log |

Escalate a level only when the current one has demonstrably failed.

## Step 2 — Write the run contract

Before the first iteration, state all of these in the thread (or a scratch file under
`.agent/scratch/`). This is the natural-language program the run executes:

- **Objective** — one sentence, measurable.
- **Mutable** — the training program and its hyperparameters. One config surface.
- **Protected** — the **evaluation code and the validation split** (changing them invalidates every
  prior comparison), `datasets/` contents, `docs/API_CONTRACT.md`, and pretrained weights pulled from
  Hugging Face.
- **Metric and direction** — the exact eval command and the number it prints, plus whether higher or
  lower is better. Record the baseline before the first change.
- **Run command** — how one iteration trains and evaluates, and how its output is parsed. See
  [`docs/TRAINING.md`](../../../docs/TRAINING.md).
- **Keep-or-revert rule** — improvement is retained as a commit; regression or crash is reverted to
  the last retained commit.
- **Crash policy** — fix if mechanical (OOM → lower batch size), else revert and record.
- **History** — where each iteration's parent commit, diff, metric, runtime, and verdict are recorded.
  This log is the actual output of the run; the weights are a by-product.
- **Escalation** — the conditions that require a human (see below).
- **Exhaustion** — what "no more ideas worth trying" looks like, so the run stops instead of churning.

## Step 3 — Declare a budget

No unattended run starts without explicit limits. State the ones that apply:

| Limit | Example |
|-------|---------|
| Experiments | 20 iterations |
| Wall-clock per run, total | 5 minutes per run, 4 hours total |
| Retries per crash | 1, then revert |
| Disk for checkpoints | cap it; checkpoints fill a drive quickly |
| GPU memory ceiling | a config that only fits by luck is not an improvement |
| Minimum evidence to retain a change | one eval run on the fixed split, recorded |

When a budget is exhausted, **return the best retained checkpoint, the experiment log, the unresolved
issues, and the reason for stopping.** Do not report a sweep's best number without saying how many
runs it took or what was skipped — see
[`verification-before-completion`](../verification-before-completion/SKILL.md).

## Step 4 — Ratchet, one change at a time

Each iteration: read the current config and recent history → propose **one** motivated change → commit
it → train → evaluate → keep or revert → record. One change per iteration is what makes the metric
attributable; batching changes destroys the signal that justifies keeping them.

Two failure modes to guard against:

- **The metric is gamed.** A ratchet improves only what it can see. A config that lowers validation
  error while doubling inference time, blowing up VRAM, or overfitting the split is not an
  improvement. Carry runtime, memory, and generalization as revert conditions, not as hopes. If the
  validation split is small, a long enough sweep will fit it — hold out a set you never tune against.
- **Judgment gets frozen into rules.** Choosing *which* change to try next is judgment and stays with
  the model — the same boundary [`/compile-skill`](../../commands/compile-skill.md) draws for
  harnesses.

## Fan-out extras

- **Define the reducer before the sweep.** Decide how configs will be ranked — and under which
  constraints — before any run starts; otherwise you get N numbers and no decision.
- **Every handoff is an artifact contract.** A worker returns the config, the metric, the runtime, and
  the checkpoint path — not prose.
- **Do not fragment coherent work.** Architecture changes and the eval harness itself are
  whole-picture decisions; do not split them across isolated workers.

## Escalate to a human

Stop and ask, regardless of remaining budget, when the run would: commit, push, or tag; upload
weights to Hugging Face or any remote; delete or rewrite dataset files; change the evaluation code or
validation split; or overwrite a released checkpoint. Publishing a model is a human decision.

## Report when the run ends

State the objective, experiments run vs. budgeted, the baseline and best metric, the retained changes
in order with their deltas, reverted attempts worth knowing about, open questions, and why the run
stopped.

## Related

- [`karpathy-guidelines`](../karpathy-guidelines/SKILL.md) — per-change discipline inside each iteration
- [`validate-implementation`](../validate-implementation/SKILL.md) — per-AC verdicts when the metric is an AC matrix
- [`/decompose`](../../commands/decompose.md) — building the independent units this contract governs

Sources: Karpathy's *autoresearch* (ratchet loop, protected eval, experiment lineage) and Anthropic's
*Building Effective Agents* (chain, route, parallelize, orchestrate, evaluate).
