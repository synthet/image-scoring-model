# Hugging Face packaging

Staging folders for published model cards (and companion dataset YAMLs). Weights are **not**
stored here — they live under `models/` locally and on Hugging Face Hub.

| Folder | Hub repo | Contents |
|--------|----------|----------|
| [`eye-pose-v0/`](eye-pose-v0/) | [synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0) | Model card + pose dataset YAML |
| [`bird-detect-v0/`](bird-detect-v0/) | [synthet/bird-detect-v0](https://huggingface.co/synthet/bird-detect-v0) | Model card + detect dataset YAML |

Upload from a staging copy that includes the `.pt` weights (never commit weights). See
[`models/README.md`](../models/README.md) for download instructions.
