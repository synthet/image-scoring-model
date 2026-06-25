# Model weights

Fine-tuned pose weights are **not** committed to git. Download the published checkpoint from Hugging Face:

**[synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0/tree/main)**

```bash
pip install huggingface_hub
huggingface-cli download synthet/eye-pose-v0 eye_pose_v0.pt --local-dir models/
```

Place `eye_pose_v0.pt` in this directory. The CLI and `PoseLocalizer` resolve it automatically.

To train your own weights from CUB-200, see [docs/TRAINING.md](../docs/TRAINING.md).
