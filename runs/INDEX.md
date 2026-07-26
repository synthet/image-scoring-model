# `runs/` — training and eval artifacts

**Gitignored.** Ultralytics (and related) run directories: weights snapshots, curves, batch previews, `.train_state.json`.

| Typical path | Semantics |
|--------------|-----------|
| `detect/wildlife_bird/` | Bird-detect fine-tune |
| `pose/` | Pose fine-tune trees |
| `eval_*.json` | Localization eval summaries |

Promote release candidates to [`models/`](../models/README.md); publish via [`hf/`](../hf/README.md). Never commit `runs/`.
