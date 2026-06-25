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

## Training a detect-only model (sketch)

1. Run the existing CUB converter to produce `data/wildlife_bird/`.
2. Strip keypoints from label `.txt` files (keep `<class> <cx> <cy> <w> <h>` only), or add a small export script.
3. Point an Ultralytics detect YAML at the same `images/train` and `images/val` paths with `names: {0: bird}`.
4. Train: `YOLO("yolo11n.pt").train(data=..., epochs=100, ...)`.

Pose fine-tuning (`training/train_pose.py`) remains the recommended path when eye localization is also required.

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

## Related docs

- [PIPELINE.md](PIPELINE.md) — how localization fits the eye-quality flow
- [TRAINING.md](TRAINING.md) — CUB bootstrap and pose fine-tuning
- [API_CONTRACT.md](API_CONTRACT.md) — current JSON output (eye boxes only)
- [BACKEND_INTEGRATION.md](BACKEND_INTEGRATION.md) — wiring eye-quality and bird_species in the backend
