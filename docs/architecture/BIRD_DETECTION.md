---
type: Technical Reference
title: Bird bounding box detection
description: Bird bbox detection, BioCLIP species crops, and detect-only training.
resource: architecture/BIRD_DETECTION.md
tags: [docs, architecture, detection]
timestamp: 2026-09-25T00:00:00Z
okf_version: 0.1
---

# Bird bounding box detection

The same CUB-200 bootstrap used for eye keypoints can produce bird bounding boxes. In fact, the **pose model already predicts a bird box** on every inference — training supervises both box and keypoints from CUB's `bounding_boxes.txt`.

## What the pose model already gives you

Each YOLO pose detection includes:

- `bbox_xyxy` — bird box in pixel coordinates `(x1, y1, x2, y2)`
- `bbox_confidence` — detection confidence
- `bbox_area_frac` — box area as a fraction of image area

Training labels encode the box as the first five fields of each YOLO pose line:

```text
<class> <cx> <cy> <w> <h> <kpt1_x> <kpt1_y> <kpt1_v> ...
```

CUB conversion reads official boxes from `bounding_boxes.txt` and passes them through `training/prepare_dataset.py` unchanged.

The eye-quality pipeline uses the bird box internally (e.g. `subject_too_small` via `subject_area_frac`) but **does not expose it in the JSON API** today. Per-eye `detections[].bbox_norm` are small squares around each eye, not the full bird.

## Getting a bird box at inference

### Option A — Reuse the fine-tuned pose model (no extra training)

```python
from eye_quality.localization.pose_model import PoseLocalizer

loc = PoseLocalizer(weights="models/eye_pose_v0.pt", device="0")
detections = loc.predict("path/to/bird.jpg")
if detections:
    best = detections[0]  # highest bbox confidence
    x1, y1, x2, y2 = best.bbox_xyxy
    conf = best.bbox_confidence
```

Normalize for API or overlay:

```python
w, h = best.image_width, best.image_height
bbox_norm = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
```

**Best when:** you need bird box and eye keypoints in one forward pass.

### Option B — Detection-only model (same data, simpler head)

Fine-tune `yolo11n.pt` (standard detect, not pose) on the same CUB images using **box labels only** — drop keypoint columns from the YOLO label files.

| Aspect | Pose (`yolo11n-pose.pt`) | Detect (`yolo11n.pt`) |
|--------|--------------------------|------------------------|
| Output | Box + 6 keypoints | Box only |
| Training loss | `box_loss` + `pose_loss` | `box_loss` only |
| Inference | Slightly heavier | Faster |
| Same CUB bboxes | Yes | Yes |

**Best when:** you only need subject localization (crop, gating, counting) and will not use eye keypoints.

### Option C — Future API field

Expose `subject_bbox_norm` and `subject_confidence` on `EyeQualityPipelineResult` by threading the best `PoseDetection` through `score_image()`. No new model required.

## Multi-bird and selection policy

- CUB has **one bird per image**; labels assume a single instance.
- YOLO can return **multiple boxes** on crowded frames. The eye pipeline picks the **highest `bbox_confidence`** detection and ignores the rest.
- For multi-bird culling, return all boxes above a confidence threshold instead of top-1.

## Caveats

| Topic | Note |
|-------|------|
| CUB box style | Tight bird boxes from dataset annotators; field photos may differ |
| Single class | Labels use class `0: bird` only |
| Habitat framing | Boxes are bird-tight, not "animal in scene" |
| Generalization | Validate on your burst library before production thresholds |

## Training a detect-only model

1. Convert CUB to the pose dataset (boxes + keypoints):

```bash
python -m training.convert_cub200 \
  --cub-root D:/Datasets/CUB_200_2011 \
  --output data/wildlife_bird \
  --val-ratio 0.15
```

2. Strip keypoints into a detect-only dataset (box labels only):

```bash
python training/make_detection_labels.py \
  --pose-dir data/wildlife_bird \
  --output data/wildlife_bird_det
```

3. Fine-tune YOLO11n detect (copies `best.pt` → `models/bird_detect_v0.pt`):

```bash
python training/train_detect.py --epochs 100 --batch 16 --device 0 \
  --output models/bird_detect_v0.pt
```

Or via `train_ctl` (pause/resume-friendly; run from repo root):

```bash
python -m training.train_ctl start --task detect --epochs 100 --batch 16 --device 0 --background
python -m training.train_ctl status --task detect
```

Config: `training/configs/wildlife_bird_det.yaml` (`names: {0: bird}`).

### Getting bbox coordinates at inference

```bash
eye-quality detect path/to/bird.jpg
# or
python -m eye_quality detect path/to/bird.jpg --weights models/bird_detect_v0.pt --device 0
```

Python API:

```python
from eye_quality.localization.bird_detector import BirdDetector

det = BirdDetector(weights="models/bird_detect_v0.pt", device="0")
birds = det.predict("path/to/bird.jpg")
if birds:
    best = birds[0]
    x1, y1, x2, y2 = best.bbox_xyxy
    bbox_norm = best.bbox_norm  # [x, y, w, h] normalized
```

Pose fine-tuning (`training/train_pose.py`) remains the recommended path when eye localization is also required.

## Teacher pseudo-labels (hard negatives, small birds)

CUB alone cannot teach small, distant or occluded birds, nor bird-free frames. The upstream open COCO
detector RTMDet-tiny (Apache-2.0) is used as a **teacher**:
- It is reimplemented in plain PyTorch in [`training/teacher/`](../../training/teacher/) and loads the
  official checkpoint with `strict=True`.
- It runs full-frame plus 2×2 tiles over a stratified library pool.
- A conservative policy produces YOLO labels: only COCO `bird` ≥ 0.50 makes a box, anything ambiguous is
  excluded, and hard negatives require no animal box ≥ 0.35 and no animal keyword.

v0 produced 877 images (503 boxes, 419 negatives), calibrated on the #377 owner labels. The #377 cohort's
folders are excluded from the pool. See
[teacher-pseudo-labels-v0-2026-09-25.md](../reports/teacher-pseudo-labels-v0-2026-09-25.md) and
[TEACHER_PSEUDO_LABELS.md](../guides/TEACHER_PSEUDO_LABELS.md).

## Species identification with BioCLIP

[BioCLIP](https://github.com/Imageomics/pybioclip) classifies images against taxonomic text prompts (zero-shot). It does **not** detect subjects — standard practice is **detect → crop → classify**, the same pattern used in camera-trap pipelines (e.g. MegaDetector → crop → BioCLIP).

### Why a bbox crop helps

On full frames, background (sky, branches, water, feeders) often dominates the embedding. When the bird is small in the frame, species accuracy drops. A bird crop from your bbox:

- Increases **subject pixel share**
- Removes **distracting context**
- Matches how BioCLIP is deployed in the field (detector crop, then classifier)

When the bird already fills most of the image, cropping adds little. For typical wildlife bursts with variable subject size, **cropping usually helps a lot**.

### Recommended flow (this repo + backend)

```text
full image (EXIF-corrected)
  → bird bbox (pose model or detect-only YOLO)     ← this repo
  → padded subject crop
  → BioCLIP species ranking                        ← backend bird_species phase
  → (optional) eye keypoints + focus score         ← this repo, same pose pass
```

One pose inference can feed **both** species ID (subject crop) and eye focus (eye keypoints). Those are separate signals; they do not need separate detectors.

### Cropping for BioCLIP

```python
from eye_quality.crop.eye_crop import load_oriented_image
from eye_quality.localization.pose_model import PoseLocalizer

oriented = load_oriented_image("path/to/bird.jpg")
detections = PoseLocalizer(weights="models/eye_pose_v0.pt").predict("path/to/bird.jpg")
if not detections:
    ...

best = detections[0]
x1, y1, x2, y2 = best.bbox_xyxy

# Optional padding (often helps CLIP-style models)
pad_frac = 0.10
bw, bh = x2 - x1, y2 - y1
x1 = max(0, x1 - pad_frac * bw)
y1 = max(0, y1 - pad_frac * bh)
x2 = min(oriented.width, x2 + pad_frac * bw)
y2 = min(oriented.height, y2 + pad_frac * bh)

crop = oriented.image.crop((int(x1), int(y1), int(x2), int(y2)))
# → pass crop to pybioclip TreeOfLifeClassifier or backend bird_species runner
```

Backend reference: [bird_species.py](https://github.com/synthet/image-scoring-backend/blob/main/modules/bird_species.py). Wire the subject crop there rather than duplicating BioCLIP in this repo unless you add a dedicated species module later.

### BioCLIP caveats

| Factor | Effect |
|--------|--------|
| **Prompt list** | Zero-shot over your species names — use scientific + common names for the region |
| **Crop tightness** | Very tight boxes can clip wings/beak; ~5–15% padding often improves results |
| **Small / distant birds** | Tiny crops still limit accuracy (independent of BioCLIP) |
| **Blur / motion** | Cropping does not fix defocus; sharp frames classify better |
| **Wrong box** | Multi-bird frames need the correct detection (top-confidence may be wrong) |
| **CUB-trained bbox** | Studio/Flickr-style boxes; validate on your field library |

### When to use detect-only vs pose for BioCLIP

| Need | Model |
|------|-------|
| Species crop + eye focus | Fine-tuned **pose** model (one pass) |
| Species crop only, fastest inference | **Detect-only** YOLO on same CUB boxes |

## Measured gap vs an open COCO detector (2026-09-24)

The backend added an arm to its owner-labelled #377 detector benchmark (339 frames). On the frames
`bird_detect_v0` misses at 640 (137 labelled birds, 71 bird-free):

| Detector | Recall | False-positive rate |
|---|---|---|
| `bird_detect_v0` @1280 | 82% | 63% |
| `bird_detect_v0` tile-on-miss | 73% | 51% |
| RTMDet-tiny (COCO, Apache-2.0) @640, animal classes ≥ 0.4 | 82% | **4%** |

On bird-free frames that `bird_detect_v0` flagged at 640, RTMDet's bird class keeps only 7/28. Where
both detect a bird, the boxes agree closely (median IoU 0.82).

Both run at about 640 px input, so the gap is **training data**. The CUB-derived set has minimum box
area 0.06, 300–500 px images and no negatives, so the model has never seen a small bird or an empty
frame.

Training priorities for the next detector version:

1. **Hard negatives** from the library: bird-free frames, especially textured foliage, bark and water,
   where the current model fires at 1280.
2. **Small subjects:** birds under 0.05 area fraction at full camera resolution. Pseudo-label them
   with an open COCO detector, and human-check a sample.
3. **Raptors on textured backgrounds** (the eagle slice). COCO calls some of them `bear`, so
   pseudo-labels must be class-agnostic ("animal").
4. Report recall and FP on the #377 strata as the acceptance test, not only mAP on CUB validation.

Full report: [backend subject-detector comparison](https://github.com/synthet/image-scoring-backend/blob/master/docs/reports/subject-detector-comparison-2026-09-24.md).

## Related docs

- [PIPELINE.md](PIPELINE.md) — how localization fits the eye-quality flow
- [TRAINING.md](../guides/TRAINING.md) — CUB bootstrap and pose fine-tuning
- [API_CONTRACT.md](../technical/API_CONTRACT.md) — current JSON output (eye boxes only)
- [BACKEND_INTEGRATION.md](../guides/BACKEND_INTEGRATION.md) — wiring eye-quality and bird_species in the backend
