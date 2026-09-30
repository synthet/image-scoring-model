---
type: Plan
title: Recall-first retraining for bird detection and eye localization
description: Data audit, independent evaluation, controlled retraining, and release gates for bird_detect_v0 and eye_pose_v0.
resource: docs/planning/bird-eye-retraining-2026-09-27.md
tags: [planning, training, detection, keypoints, evaluation]
timestamp: 2026-09-27T00:00:00Z
okf_version: 0.1
status: proposed
---

# Recall-first bird and eye retraining

User priority: **missing birds or visible eyes is worse than false detections**. Optimize recall under an explicit false-positive budget. This is an implementation plan; no training or checkpoint promotion has been performed.

The first deliverable is a better bird detector using existing field data. The second is a better eye localizer using corrected labels and human-annotated field examples. Preserve the existing output schema, six-keypoint order, and score ranges. Retraining these models improves localization; it does not directly train the heuristic focus score.

Use the backend's LLM CLI judge workflow as required quality control: [judge protocol and integration plan](llm-cli-retraining-judges.md). Two qualified independent vision judges review blind image packets; a third handles disagreements. Explicitly inspect model misses, audit eye visibility, and hold candidate promotion on unresolved critical findings. Keep agent-derived review evidence separate from human benchmark labels.

## Evidence and starting point

Local inspection on 2026-09-27:

| Item | Finding | Implication |
|---|---|---|
| CUB pose and detect labels | Each has 10,018 training and 1,752 validation label files; none empty | Add real bird-free scenes and small field birds |
| `data/pseudo_rtmdet_v0/` | 803 training labels, including 385 empty labels; 74 validation labels, including 34 empty labels | Existing pilot data can support the first detect experiment |
| Hardware | RTX 4060 Laptop GPU, 8,188 MiB | Start with nano models; measure batch size and throughput before a run budget |
| Existing run logs | Detect and pose logs reach epoch 100 at 640 px | More epochs alone are not the first intervention |
| Existing training scripts | Fixed run name, `exist_ok=True`, output defaults to released v0 paths | Add isolated run paths before launching experiments |
| Existing localization evaluation | Reports coverage/confidence without ground-truth matching | Cannot establish eye accuracy or false-positive rate |
| Eye crop data | `eye_dataset` has a manifest with model-generated coordinates; sampled record has no grade; `eye_dataset_val/manifest.jsonl` is empty | Audit all annotations; these are not established human ground truth |

Checkpoint identity, to use for all baseline comparisons:

```text
models/bird_detect_v0.pt
sha256 e493c11f6799789eddeec46f394361d6831bb6d9e927a43c0f135b81f668e739
models/eye_pose_v0.pt
sha256 ec9e4b80a93eb6e2b21633e2ff2df30c49faa1eb7919dd24345c7dc295b2d1be
```

Files were hashed, not loaded for inference. Existing CSV metrics have not been proven to belong to these exact checkpoint hashes. Re-evaluate both rather than treating historical logs as their baseline.

The [detector report](../architecture/BIRD_DETECTION.md) records 82% recall with 63% false positives for v0 at 1280 on a selected difficult cohort, versus 82%/4% for the teacher's animal detections. These are historical, subset-specific results with different class policies, not full-library accuracy or a directly interchangeable bird-only target.

The [teacher report](../reports/teacher-pseudo-labels-v0-2026-09-25.md) confirms that #377 images and folders were excluded from training, but their labels were used to select teacher thresholds. Treat #377 as a development/regression benchmark, **not an untouched final test**. Its 339-frame cohort and 337 owner-labelled frames also require explicit eligibility accounting.

## 1. Freeze evaluation before changing training

Deliver a versioned evaluation manifest, annotation guide, evaluator, and v0 baseline report in a separate change before claiming model improvements.

- Preserve existing CUB validation membership. Create a new field calibration set and a new final test from previously unused folders/sessions; separate by burst, source image, and near-duplicate group. Keep derivatives of one source in the same split. Exclude both sets from teacher pools, active learning, and crop pretraining.
- Proposed initial annotation budget: 800-1,200 field frames divided between calibration and final test, approximately half bird-positive and half verified bird-free. Include enough independent sessions; hundreds of adjacent frames do not provide hundreds of independent examples. Keep enriched challenge slices separate from prevalence-weighted production estimates.
- Label all birds, bounding boxes, visible eye centers, eye visibility, and whether the eye is actually resolvable. Include small subjects, raptors, silhouettes, occlusion, flight, textured backgrounds, multiple birds, and other animals. Record intended subject separately for multi-bird culling.
- Human-review negatives at useful resolution. Teacher silence and absent keywords do not prove that a frame is bird-free. Ambiguous frames stay excluded pending review.
- Fix preprocessing: EXIF orientation, RAW rendition, resizing, crop transforms, and coordinate spaces. Compare models on identical decoded images and input settings.
- Save raw low-threshold predictions once; tune bird and keypoint thresholds on calibration only. Freeze thresholds before final testing. Compare both unchanged inference settings and separately calibrated operating points.

| Metric | Definition and purpose |
|---|---|
| Bird instance recall | One-to-one same-class matches at IoU >= 0.5 divided by all annotated birds; missed instances count even in a frame with another detected bird |
| Bird frame recall | Bird-positive frames with at least one correct matched bird / eligible positive frames |
| Bird-free frame FP rate | Verified bird-free frames with any accepted bird detection / verified bird-free frames |
| Precision and FP/image | Count unmatched/duplicate boxes, including false boxes on bird-positive images |
| Visible-eye recall | Matched visible eyes within a frozen tolerance / all annotated resolvable visible eyes; upstream bird misses remain failures |
| Eye precision | Correct accepted visible-eye localizations / all accepted visible-eye localizations; include hidden/turned-away-eye hallucinations |
| Eye error | Median and p90 center error in original pixels and normalized by human-annotated head size; freeze tolerance from crop needs and annotator agreement |
| Downstream utility | Correct-eye crop coverage, wrong-subject rate, and focus ranking on independently reviewed bursts |
| Runtime | Median/p95 latency and peak VRAM with decoding included and excluded, at matched hardware/settings |

Report denominators, per-size and per-condition results, and session/burst-clustered confidence intervals. Report conditional eye accuracy given a correctly detected bird separately from end-to-end eye recall. mAP remains diagnostic; it is not the release decision. Cross-model eye agreement is not ground-truth accuracy.

**Proposed gates to freeze after the baseline, before training:** maximize bird recall at <=10% bird-free frame FP rate, with <=5% as a secondary operating point; also report recall at v0's measured FP rate. Aim for >=25% relative reduction in bird misses and >=20% relative reduction in visible-eye misses versus calibrated v0, with eye precision >=95%. These are targets, not expected results. Require no material regression in large-bird/CUB performance or critical field slices, and report uncertainty when sample sizes cannot establish a gain. Revisit a target only as a new experiment contract, not after viewing final-test results.

## 2. Correct supervision and make runs reproducible

### Eye labels

In [convert_cub200.py](../../training/convert_cub200.py), `_infer_bilateral_eyes` can generate both eye positions from forehead/crown geometry when neither eye is visible, and return them with a positive visibility flag. This is a concrete potential source of false eye supervision; its contribution to the released checkpoint is not yet measured.

Audit against original CUB parts, count affected examples, and create a new training dataset version. Preserve genuine coordinates; never mark guessed eye locations as observed. Map CUB's visible flag to the documented target convention explicitly. Retain bird boxes when eyes are hidden rather than dropping such birds. Keep validation labels and evaluator frozen for a model comparison; if annotation corrections affect evaluation, make a separate benchmark revision and rebaseline every model on it.

Ultralytics pose loss uses `v != 0` as the keypoint mask, so v=1 and v=2 do not create separate visible-versus-occluded classes. Keypoint confidence is not a calibrated probability that an eye is visible. Verify the installed version before implementing visibility handling. Keep human visibility metadata for evaluation; any dedicated visibility head is a separate experiment. [Upstream loss implementation](https://docs.ultralytics.com/reference/utils/loss/), [pose label format](https://docs.ultralytics.com/datasets/pose/).

**Do not pad bird-box-only pseudo-labels with six zero keypoints and mix them into ordinary pose training.** That treats unknown keypoints as negative objectness targets. Use those labels for detect training first. Adding them to pose training requires an explicit per-instance annotation mask that suppresses both keypoint coordinate and objectness losses for unknown annotations, or human pose labels. Truly hidden eyes and unannotated eyes are different states.

### Run controls

Extend [train_detect.py](../../training/train_detect.py) and [train_pose.py](../../training/train_pose.py) with configurable project/name, seed, patience, optimizer/LR, and augmentation controls. These are proposed additions, not existing CLI flags. Default new experiments to unique directories with overwrite protection; require an explicit candidate output away from v0 paths. Persist dataset hashes, starting checkpoint hash, package versions, GPU, resolved arguments, best/last checkpoints, metrics, and runtime.

Fine-tuning uses v0 as initialization with a fresh optimizer; resume is only for continuing the same interrupted experiment. Before unattended execution, write the [autonomous run contract](../../.cursor/skills/autonomous-run-contract/SKILL.md) with epoch/time limits, stop criteria, checkpoints, and recovery. Fix the CUDA Python environment first; this inspection found no repo `.venv`, and the default `python` launcher failed.

## 3. Bird detector: existing data first, then targeted labels

1. Review the existing pseudo-labels across all strata. Prioritize all 385 training negatives if feasible, plus small positives and multi-bird completeness. Preserve its existing validation separation. Create a new corrected dataset version rather than rematerializing over v0.
2. Combine CUB training positives with reviewed field positives and negatives using explicit sampling weights. Simply appending 803 frames to 10,018 CUB frames would leave field data underrepresented. Pilot approximately 50% CUB positives, 35% field positives, and 15% verified negatives per epoch; treat this as one initial recipe, not a claimed optimum. Use augmentation without allowing a handful of bursts to dominate.
3. Human-label 300-500 additional difficult positive training frames, prioritizing the teacher's uncertain/missed birds, occluded raptors, tiny birds, and student/teacher disagreements. Label every bird in each selected frame. Never relabel a confident mammal as a bird based only on a noisy keyword; the teacher report documents that failure.
4. Preserve full-frame small-object examples. Add crop/scale augmentation with correct labels; verify that augmentation does not erase already tiny subjects. Synthetic downscaling supplements real small birds, not replaces them.

| Experiment | Controlled change | Budget / decision |
|---|---|---|
| D0 | Frozen v0, confidence sweep only | Establish what calibration can recover without training |
| D1 | v0 initialization; CUB + reviewed existing field data; 640 | 3-epoch smoke, then up to 40 epochs, patience 10 |
| D2 | D1 recipe plus additional human hard positives | Same settings; isolate the annotation benefit |
| D3 | Best data recipe at 960 | Compare recall/latency; try 1280 only if size-slice evidence and VRAM justify it |
| D4, conditional | Same best recipe initialized from COCO base, or a small model | One variable per run; only if earlier runs plateau |

On the 8 GB GPU, pilot batch 8 at 640 and batch 2-4 at 960, with AMP; lower after measured OOM. These are starting estimates, not guaranteed fits. Confirm the winning recipe with two additional seeds. Do not launch a broad sweep. Background examples are an established way to reduce false positives; the mixture still needs field validation. [Ultralytics training guidance](https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results/).

## 4. Eye pose: corrected targets, field labels, and coverage

Allocate an initial 500-1,000 training frames with real eye/head annotations, independent of calibration/test. Preserve six-keypoint semantics and horizontal flip mapping. Include profiles with only one visible eye, turned-away birds, tiny/unresolvable eyes, closed eyes, occlusion, and motion blur. Verify both sides manually on a sample. Box teachers do not supply eye ground truth.

| Experiment | Controlled change | Decision |
|---|---|---|
| P0 | Frozen v0 and threshold sweep | Separate bird misses, eye misses, wrong position, wrong subject, and visibility mistakes |
| P1 | v0 initialization on corrected CUB training labels | Isolate label correction; up to 40 epochs after smoke |
| P2 | P1 recipe plus human field pose labels and reviewed bird-free frames | Prioritize end-to-end visible-eye recall under eye-precision gate |
| P3, conditional | Detection followed by padded bird-crop pose inference/training | Test if input resolution or upstream detection still limits recall |

For P3 first compare human-box crops (upper-bound diagnostic) with detector crops (deployable behavior). Map points back to the original oriented frame; include imperfect/jittered boxes during training. Count detector misses in end-to-end metrics. Crop-based inference is a pipeline change that needs its own latency and contract checks, not a free weight replacement. Keep single-pass pose as the default comparison.

Audit inference policies separately from weight experiments: `localize_eyes` selects only the highest-confidence bird; lowering thresholds can increase wrong-subject selections. The current `TOO_SMALL` branch compares a crop size against a minimum already enforced by `max`, making that branch unreachable for ordinary finite inputs. These need isolated behavior tests and separate evaluation, rather than attributing all errors to training.

Defer [self-supervised landmark pretraining](ssl-landmark-pretraining.md) and [eye-location density work](eye-location-density.md) until these simpler experiments show the remaining limitation. Keep focus scoring fixed while comparing localization changes; localization confidence must not become a focus score.

## 5. Implementation sequence and verification

Proposed files below are additions unless already linked above. Each stage is independently reviewable.

| Stage | Files / areas | Acceptance checks before proceeding |
|---|---|---|
| A: evaluation and baseline | `training/eval_detection.py`, `training/eval_eye_accuracy.py`, `training/audit_training_data.py`; versioned manifests under ignored `data/` | Freeze matching, thresholds, exclusions, and v0 results; no metric-win claim |
| B: data/run plumbing | `training/convert_cub200.py`, `training/prepare_dataset.py`, both train scripts; new versioned dataset YAMLs | Immutable splits; no false inferred eye targets; no overwrite of v0 or prior runs |
| B2: CLI judge control | `training/review/` packet builder, schema, consensus and shared CLI transport adapter; [judge protocol](llm-cli-retraining-judges.md) | 24-item qualification smoke; image-access controls; independent votes; explicit abstentions and human audit queue |
| C: detect pilot | Reviewed mixed dataset, D1/D2 run contracts and reports | Calibrated recall/FP gate and independently annotated box QA |
| D: pose pilot | Human field annotations, P1/P2 contracts and reports | End-to-end eye recall/precision and crop accuracy gates |
| E: final comparison | Frozen evaluator, final holdout, matched runtime measurements | Pick candidate once; prepare promotion/rollback evidence |

Meaningful planned assertions for the fast subset (write before implementing their corresponding behavior):

- `tests/test_training_data_audit.py`: `test_burst_and_derivative_split_leakage_rejected`, `test_unreviewed_negative_not_accepted`.
- `tests/test_convert_cub200.py`: `test_head_only_annotation_does_not_create_visible_eyes`, `test_hidden_eye_bird_retains_bbox`, `test_visible_eye_coordinates_preserved`.
- `tests/test_training_runs.py`: `test_released_output_rejected`, `test_existing_run_not_overwritten`, `test_fresh_finetune_does_not_resume_optimizer`.
- `tests/test_eval_detection.py`: `test_duplicate_detection_counts_as_false_positive`, `test_multi_bird_miss_counts_as_false_negative`, `test_empty_frame_fp_denominator`.
- `tests/test_eval_eye_accuracy.py`: `test_upstream_miss_counts_against_eye_recall`, `test_hidden_eye_prediction_is_false_positive`, `test_out_of_tolerance_eye_counts_fp_and_fn`.
- If P3 proceeds: coordinate round-trip/orientation tests, and a GPU test for actual candidate crop inference.

Run the **fast subset** (`python -m pytest -m "not gpu"`) for logic and **GPU tests** (`python -m pytest -m gpu`) for actual weights/CUDA integration. Neither substitutes for the frozen field benchmark. For each experimental model report both kinds separately, plus whether the candidate itself was exercised; existing GPU tests may only cover default v0.

## Completion and rollback

Deliver `bird_detect_v1_candidate` and `eye_pose_v1_candidate` as run-scoped artifacts with hashes, model cards, dataset provenance, calibrated thresholds, error galleries, per-slice metrics, confidence intervals, runtime, and the CLI judge consensus/audit report with critical disputes resolved. A detector improvement can ship independently if the pose candidate fails its gates.

Keep v0 files untouched and retain explicit model-path selection for A/B comparison and rollback. Promotion, weight publication, and any output-contract change remain human decisions under [SAFETY.md](../../.agent/SAFETY.md). If a final test fails, document it and begin another development cycle; do not repeatedly tune against that holdout while calling it independent.

First executable milestone: freeze the field evaluator and baseline, audit the existing 803-image training pilot, then run **one isolated D1 experiment**. Set wall-clock estimates from its measured throughput; annotation availability is likely the larger uncertainty.
