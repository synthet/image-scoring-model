# Model weights

Fine-tuned weights are **not** committed to git. Download published checkpoints from Hugging Face:

## Pose (eye localization)

**[synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0/tree/main)**

```bash
pip install huggingface_hub
huggingface-cli download synthet/eye-pose-v0 eye_pose_v0.pt --local-dir models/
```

Place `eye_pose_v0.pt` in this directory. The CLI and `PoseLocalizer` resolve it automatically.

## Detect (bird bounding boxes)

**[synthet/bird-detect-v0](https://huggingface.co/synthet/bird-detect-v0/tree/main)**

```bash
huggingface-cli download synthet/bird-detect-v0 bird_detect_v0.pt --local-dir models/
```

Place `bird_detect_v0.pt` here. Then:

```bash
python -m eye_quality detect path/to/bird.jpg --weights models/bird_detect_v0.pt
```

To train your own weights from CUB-200, see [docs/guides/TRAINING.md](../docs/guides/TRAINING.md).

## Publishing a checkpoint

Publishing weights is a **human decision** (see [`.agent/SAFETY.md`](../.agent/SAFETY.md)). Live
repos: [synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0) and
[synthet/bird-detect-v0](https://huggingface.co/synthet/bird-detect-v0).

1. Read `HF_TOKEN` from `.env` (never echo, log, or commit it).
2. Stage a model card (`README.md` with YAML frontmatter) plus the `.pt` under a copy of
   [`hf/eye-pose-v0/`](../hf/eye-pose-v0/) or [`hf/bird-detect-v0/`](../hf/bird-detect-v0/) (temp dir is fine).
3. Upload with `HfApi.upload_folder` (or `huggingface-cli upload`).
4. Delete the staged `.pt` so it cannot be committed; keep only the card (and optional YAML) in `hf/`.

Hugging Face model cards for publishing live under [`hf/`](../hf/README.md).
