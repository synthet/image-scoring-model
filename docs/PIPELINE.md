# Pipeline

The eye-quality pipeline scores wildlife images for eye focus. It is designed for **stack-relative culling**: prefer frames with sharper eyes within a burst, and flag technically weak images as a secondary signal.

## Stages

```text
image load (EXIF-oriented)
    → YOLO pose localization (6 head keypoints)
    → native-resolution eye crops
    → per-eye heuristic metrics
    → aggregate to image-level summary
```

### 1. Localization

`PoseLocalizer` runs Ultralytics YOLO11n-pose with fine-tuned bird weights (`models/eye_pose_v0.pt` when present). Download from [synthet/eye-pose-v0 on Hugging Face](https://huggingface.co/synthet/eye-pose-v0/tree/main).

**Keypoint schema** (6 points, index order matters for training labels):

| Index | Name | Role |
|-------|------|------|
| 0 | `beak` | Head orientation |
| 1 | `left_eye` | Eye crop anchor |
| 2 | `right_eye` | Eye crop anchor |
| 3 | `head_top` | Head width estimate |
| 4 | `left_shoulder` | Body context |
| 5 | `right_shoulder` | Body context |

Horizontal flip augmentation swaps indices `[0, 2, 1, 3, 5, 4]`.

Detections are filtered by bounding-box confidence, area fraction, and per-keypoint confidence (`KeypointThresholds` in `src/eye_quality/localization/keypoint_schema.py`). The highest-confidence bird box is selected when multiple detections are returned. Each visible eye becomes an `EyeLocalization` with normalized bbox, visibility class, and head-width estimate for crop sizing.

The pose model predicts a **full bird bounding box** alongside keypoints (supervised from CUB `bounding_boxes.txt` during training). That box is used for `subject_too_small` but is not yet exposed in the JSON API. The same box supports **padded subject crops for BioCLIP species ID** in the backend. See [BIRD_DETECTION.md](BIRD_DETECTION.md).

### 2. Eye cropping

`crop_eye_region` extracts a **square crop at native resolution** centered on each eye keypoint. Crop size scales with estimated head width (not a fixed pixel box). EXIF orientation is applied before cropping so coordinates match the displayed image.

Optional `--debug-dir` writes crop JPEGs for visual inspection.

### 3. Heuristic scoring

Per-eye metrics from classical image analysis (`src/eye_quality/scoring/heuristics.py`):

| Metric | Method |
|--------|--------|
| Sharpness | Laplacian variance |
| Edge energy | Tenengrad (Sobel) |
| Clarity | Local RMS contrast |
| Noise penalty | High-frequency residual estimate |
| Size factor | Penalty when crop is too small |

OpenCV is used when available; NumPy fallbacks handle platforms where specific `cv2` calls fail.

### 4. Aggregation

`aggregate_eye_quality` selects the **best visible eye** (highest focus score among eyes above confidence and size gates), then produces the image-level `EyeQualitySummary`.

**Focus score** fuses sharpness, clarity, and edge energy, scaled by size factor and noise penalty.

**Quality grade** (0–4):

| Grade | Focus score |
|-------|-------------|
| 4 | ≥ 0.85 |
| 3 | ≥ 0.65 |
| 2 | ≥ 0.45 |
| 1 | ≥ 0.20 |
| 0 | < 0.20 |

**Failure types** are confidence-gated: low-confidence detections yield `failure_type: none` rather than a hard reject. See [API_CONTRACT.md](API_CONTRACT.md) for the full enum list.

## Python API

```python
from eye_quality import score_image
from eye_quality.pipeline import PipelineConfig

config = PipelineConfig(
    weights="models/eye_pose_v0.pt",  # optional; auto-resolved if omitted
    debug_dir="/tmp/eye-debug",
)
result = score_image("path/to/bird.jpg", config=config)
payload = result.to_api_dict()
```

`localize_fn` can be injected for unit tests without loading YOLO weights.

## CLI

```bash
eye-quality score path/to/image.jpg [--weights PATH] [--debug-dir DIR] [--output out.json]
eye-quality batch manifest.jsonl --output results.jsonl
```

Manifest format: one JSON object per line with a `path` field pointing to an image file.

## MVP roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| MVP 1 | **Current** | YOLO pose localization + heuristic eye sharpness |
| MVP 2 | Planned | Learned eye-quality classifier on labeled crops |
| MVP 3 | Planned | Stack-aware ranker for burst culling |
