# Backend and gallery integration (phase 2)

This document describes how to wire **eye-quality** into **image-scoring-backend** and **image-scoring-gallery** after the model repo MVP is stable. Canonical backend docs remain the source of truth — do not duplicate them here.

Related docs: [PIPELINE.md](PIPELINE.md), [API_CONTRACT.md](API_CONTRACT.md), [TRAINING.md](TRAINING.md), [BIRD_DETECTION.md](BIRD_DETECTION.md).

## Prerequisites

- Install this package in the backend environment: `pip install -e /path/to/image-scoring-model`
- Fine-tuned pose weights at `models/eye_pose_v0.pt` — download [synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0/tree/main) or see [TRAINING.md](TRAINING.md)
- Wildlife keyword gating (`wildlife`, `birds`, `animals`) similar to [bird_species](https://github.com/synthet/image-scoring-backend/blob/main/modules/bird_species.py)

## Backend checklist

### 1. Pipeline phase

| Item | Location |
|------|----------|
| Add `PhaseCode.EYE_QUALITY = "eye_quality"` | `modules/phases.py` |
| Seed phase (optional, after keywords, sort_order ~55) | `SEED_PHASES` |
| Prerequisites | `keywords` or `scoring` complete |
| Runner | `modules/eye_quality_runner.py` |
| Register executor | `modules/phase_executors.py` |
| Job dispatcher alias | `modules/job_dispatcher.py` |
| Phase policy completeness | `modules/phases_policy.py` |
| Frontend stage list | `frontend/src/constants/pipeline.ts` |

Runner contract matches `BirdSpeciesRunner`: `start_batch`, `get_status`, `stop`, background thread, IPS updates.

### 2. Call this package

```python
from eye_quality import score_image
from eye_quality.pipeline import PipelineConfig

config = PipelineConfig(weights="models/eye_pose_v0.pt")
result = score_image(jpeg_path, config=config)
payload = result.to_api_dict()
```

For RAW files, resolve a **full embedded preview JPEG** before calling (not 512px thumbnails).

### 2b. Bird bbox crop → BioCLIP (bird_species phase)

The existing **bird_species** phase classifies species (typically via BioCLIP). Feeding a **subject crop** from the pose model bbox improves accuracy vs full-frame when the bird is small in the image.

Suggested orchestration:

1. Run pose inference once (this package or shared weights).
2. Crop the oriented image to `bbox_xyxy` with ~10% padding.
3. Pass the crop to `bird_species` / BioCLIP (backend).
4. Optionally run `score_image()` on the same file for eye focus (reuses the same weights).

Details and crop code: [BIRD_DETECTION.md](BIRD_DETECTION.md#species-identification-with-bioclip). Avoid running two separate detectors unless pose weights are unavailable.

### 3. Database (Alembic in backend)

**Aggregate table** (one row per image):

```sql
CREATE TABLE image_eye_quality (
    image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    best_eye_side TEXT,
    eye_visible_count INTEGER NOT NULL DEFAULT 0,
    focus_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    sharpness_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    clarity_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    quality_grade INTEGER NOT NULL DEFAULT 0,
    failure_type TEXT NOT NULL DEFAULT 'none',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);
```

**Optional per-eye debug** (`image_eye_detections`): bbox, visibility, crop_path, detection_confidence.

**Gallery sort**: dual-write `focus_score` to `image_model_scores` with `model_name = 'eye_quality'`.

### 4. API

Add detail-only block on `GET /api/images/{id}`:

```json
{
  "eye_quality": { "...": "same shape as EyeQualityPipelineResult.eye_quality" }
}
```

Full field reference: [API_CONTRACT.md](API_CONTRACT.md). Pattern: `technical_failure_detection` in `modules/api.py` + `get_image_technical_failure()`.

### 5. Scoring fusion (optional)

For wildlife stacks:

```text
technical_score_final = technical_score_base * 0.75 + eye_focus_score * 0.25
```

Only when `eye_visible_count > 0` and `confidence >= min_confidence`. Never auto-reject on low confidence.

### 6. Docs to update (backend)

- `docs/CANONICAL_SOURCES.md`
- `docs/technical/API_CONTRACT.md`
- `docs/reference/api/openapi.yaml`
- `docs/technical/PIPELINE_TERMINOLOGY.md`
- `docs/technical/DB_SCHEMA.md`
- `docs/IMAGE_PIPELINE.md`
- `docs/log.md`

Regenerate `openapi.json` and sync gallery `api-contract/openapi.json`.

## Gallery checklist

| Item | File |
|------|------|
| Register model label | `electron/scoringModels.ts` |
| Sort by `model:eye_quality` | already supported generically |
| Viewer ScoreBar | `src/components/Viewer/ImageViewer.tsx` |
| Optional stack sort highlight | `electron/db.ts` / culling analytics |
| Regenerate API types | `npm run generate:api-types` |

Display example:

```text
Eye Focus: 82%
Eye Quality: Good (grade 3)
Failure: none
Confidence: 91%
```

## Regression tests (backend)

- No eye found → grade 0, no reject
- One / two eyes → best eye selected
- Tiny subject → `subject_too_small`, no auto-delete
- Low confidence → `failure_type=none`
- EXIF rotation stable crop coords
- Stack with no picked images → no auto-remove

## Verification

```bash
# Model repo
pytest -m "not gpu"

# Backend (after integration)
python scripts/doctor.py --no-gpu
python -m pytest -m "not gpu and not db and not ml and not firebird"

# Gallery
npx tsc --noEmit
npm run lint
```
