# API contract

Stable JSON shape produced by `score_image()` and the `eye-quality` CLI. Backend integration should treat this as the canonical payload (see [BACKEND_INTEGRATION.md](BACKEND_INTEGRATION.md)).

## Top-level structure

```json
{
  "eye_quality": { },
  "detections": [ ],
  "debug": { }
}
```

Serialize with `result.to_api_dict()` or `result.model_dump(mode="json")`.

## `eye_quality` (image summary)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `best_eye_side` | `"left"` \| `"right"` \| `null` | | Side used for the aggregate score |
| `eye_visible_count` | int | 0–2 | Eyes above visibility threshold |
| `focus_score` | float | 0–1 | Primary ranking signal |
| `sharpness_score` | float | 0–1 | Best-eye Laplacian-based sharpness |
| `clarity_score` | float | 0–1 | Best-eye local contrast |
| `quality_grade` | int | 0–4 | Discrete grade from focus score |
| `failure_type` | string | see below | Technical weakness flag |
| `confidence` | float | 0–1 | Gated localization confidence |
| `model_name` | string | | e.g. `eye_quality_heuristic` |
| `model_version` | string | | e.g. `0.1.0` |

### Quality grades

| Grade | Meaning | Focus score |
|-------|---------|-------------|
| 0 | Unusable / no eye | < 0.20 |
| 1 | Soft | ≥ 0.20 |
| 2 | Acceptable | ≥ 0.45 |
| 3 | Good | ≥ 0.65 |
| 4 | Excellent | ≥ 0.85 |

### `failure_type` values

| Value | When assigned |
|-------|---------------|
| `none` | No failure, or confidence too low to assert one |
| `defocus` | Low sharpness with adequate size |
| `motion_blur` | Directional blur pattern |
| `subject_motion` | Combined motion indicators |
| `noise` | High noise penalty |
| `low_contrast` | Flat local contrast |
| `overexposed` | Blown highlights in crop |
| `hidden_eye` | Eye not visible |
| `turned_away` | Head turned from camera |
| `subject_too_small` | Bird bbox or eye crop below minimum |

Low-confidence results should **not** trigger auto-reject; `failure_type` stays `none`.

## Bird subject box (not in API yet)

The pose model also predicts a **full bird bounding box** per detection. It is used internally for `subject_too_small` but is **not** included in `to_api_dict()` output today. Per-eye `bbox_norm` values below are small squares around each eye, not the bird subject.

To access the bird box now, call `PoseLocalizer.predict()` directly. Pad the crop before sending to BioCLIP for species ID (see [BIRD_DETECTION.md](BIRD_DETECTION.md#species-identification-with-bioclip)). A future field such as `subject_bbox_norm` may be added to the pipeline result.

## `detections` (per eye)

| Field | Type | Description |
|-------|------|-------------|
| `eye_side` | `"left"` \| `"right"` | |
| `bbox_norm` | `[x, y, w, h]` | Normalized **eye** box in oriented image space |
| `visibility` | string | see below |
| `detection_confidence` | float | Keypoint + bbox confidence |
| `crop_path` | string \| `null` | Debug crop path when `--debug-dir` set |

### `visibility` values

`visible`, `partially_visible`, `hidden`, `turned_away`, `too_small`, `occluded`

## `debug`

| Field | Type | Description |
|-------|------|-------------|
| `crop_paths` | string[] | All exported debug crop files |
| `pose_keypoints` | float[][] | Raw keypoint rows for inspection |
| `notes` | string[] | Pipeline messages (e.g. no detection, fallback) |

## Example

```json
{
  "eye_quality": {
    "best_eye_side": "left",
    "eye_visible_count": 1,
    "focus_score": 0.72,
    "sharpness_score": 0.68,
    "clarity_score": 0.75,
    "quality_grade": 3,
    "failure_type": "none",
    "confidence": 0.88,
    "model_name": "eye_quality_heuristic",
    "model_version": "0.1.0"
  },
  "detections": [
    {
      "eye_side": "left",
      "bbox_norm": [0.42, 0.31, 0.04, 0.04],
      "visibility": "visible",
      "detection_confidence": 0.88,
      "crop_path": "/tmp/eye-debug/bird_left.jpg"
    }
  ],
  "debug": {
    "crop_paths": ["/tmp/eye-debug/bird_left.jpg"],
    "pose_keypoints": [],
    "notes": []
  }
}
```

## Batch output

`eye-quality batch` writes one JSON object per line (JSONL), each matching the structure above, with an added `path` field for the source image.
