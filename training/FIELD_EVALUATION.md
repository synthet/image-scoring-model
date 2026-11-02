# Field bird and eye evaluation

`training/eval_field.py` compares frozen, human-reviewed annotations with exported
predictions. It does **not** run a model or certify a checkpoint. Keep private photo
paths, annotations, predictions, and reports under gitignored `data/` or another
private directory. Do not use a development set as the final test after tuning on it.

## Ground-truth JSONL

One row per image, with `schema_version: 1`. Coordinates are pixels in the exact
display-oriented rendition given by `image_path`. `rendition_sha256` is the SHA-256
of that rendition's **file bytes**; predictions must name the same hash. Use one
`group_id` for every frame in a burst and for derivatives of the same source.
Groups cannot cross `calibration` and `test`. All birds in a positive frame must be
boxed. An empty `birds` list is permitted only when a human verified the frame is
bird-free; `reviewed` must be `true` on every row.
The validator also rejects an identical rendition hash on both sides of the split.

```json
{"schema_version":1,"image_id":"frame-1","image_path":"D:/private/renditions/frame-1.jpg","group_id":"shoot-12-burst-4","split":"test","pixel_space":"display_oriented","rendition_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","width":100,"height":100,"reviewed":true,"birds":[{"id":"bird-1","bbox":[10,10,50,50],"eyes":{"left":{"state":"visible","x":22,"y":22},"right":{"state":"hidden"}}}]}
```

Eye labels are optional for box-only frames. To evaluate eyes on a bird-containing
frame, annotate **both anatomical sides of every bird**. Each side's state is
`visible`, `hidden`, `unresolvable`, or `unknown`. Only `visible` has `x` and `y`.
`unknown` means the reviewer could not establish that eye's state; it is excluded
from eye scoring. A frame with any box-only bird is excluded from eye metrics, while
remaining in bird metrics. `unresolvable` means the eye could not be located at
this rendition's resolution. Confirm visibility at original resolution before
marking a tiny bird `hidden` or a frame bird-free.

## Prediction JSONL

One row is required for **every** annotated image in the selected split, including
frames with no detections. Export raw per-image model results on the same rendition;
the evaluator applies the frozen confidence thresholds. Eye keys may be omitted
for a box-only detector.

```json
{"image_id":"frame-1","rendition_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","detections":[{"bbox":[10,10,50,50],"confidence":0.91,"eyes":{"left":{"x":22,"y":22,"confidence":0.88}}}]}
```

Run from the repository root with a Python environment; this evaluator itself
uses only the standard library:

```powershell
python -m training.eval_field data/field_v1/manifest.jsonl data/field_v1/predictions_v1.jsonl `
  --split calibration --bird-conf 0.25 --eye-conf 0.8 --iou 0.5 `
  --eye-tolerance-frac 0.02 --output data/field_v1/calibration_v1.json
```

Freeze the manifest, matching rules, confidence thresholds, and eye tolerance
before scoring `--split test`. The test split must come from different folders,
bursts, and source images. Record hashes of the manifest, predictions, weights,
and evaluator commit in each experiment report. The evaluator checks rendition
identity and group separation; it cannot prove that an annotator found every
bird or that two different group IDs are unrelated.

## Metrics and limits

- Bird boxes use one-to-one maximum-cardinality matching at the configured IoU.
  Extra boxes, including duplicates on a correctly found bird, are false positives.
- Bird instance recall counts **all** annotated birds. Bird frame recall requires
  at least one correct match. Bird-free frame FP rate uses only reviewed empty frames.
- Eye recall is end-to-end: an upstream bird miss also misses every reviewed
  visible eye on that bird. A prediction on a hidden eye is a false positive.
  A point outside tolerance counts both one false positive and one false negative.
- Eye tolerance is a fraction of the human bird-box diagonal, chosen and frozen
  on calibration. Reports also give median and p90 pixel error for predicted
  visible eye pairs. Null metrics mean there was no eligible denominator.
- A box meeting IoU does not prove that its crop is useful for species or focus.
  Human primary-crop usability grades and downstream burst ranking must be
  reported separately. Confidence intervals should be clustered by group in
  the experiment report; aggregate point estimates alone do not establish a win.
