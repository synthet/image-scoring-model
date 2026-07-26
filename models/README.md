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
