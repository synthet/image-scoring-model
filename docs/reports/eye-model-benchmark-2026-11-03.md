---
type: Report
title: Eye model benchmark on CUB and a labelled library sample
description: Findings from comparing our YOLO eye models, the pipeline's crop-path eyes and an external top-down bird-eye model on CUB held-out species and 147 owner-labelled library frames.
resource: docs/reports/eye-model-benchmark-2026-11-03.md
tags: [report, eye-pose, evaluation, benchmark, field-labels]
timestamp: 2026-11-02T00:00:00Z
okf_version: 0.1
status: complete
---

# Eye model benchmark (2026-11-02)

We compared our eye localizers, and the crop-path eyes that the scoring pipeline stores for the fleet, with an **external top-down bird-eye model** that runs on a crop of a bird box. The comparison covered CUB held-out species and a library sample of 147 frames that the owner click-labelled. The work was done on 2026-11-02 local time; the file name carries the UTC date.

The external model is a private teacher. Its identity, lineage and local paths are recorded only under gitignored `docs/private/` ([PRIVATE_LOCAL.md](../PRIVATE_LOCAL.md)). The owner granted training use on 2026-11-03 (recorded in the private manifest); the training gates in the T-series pilot audit still apply.

## Summary

1. **Eye placement:** `bird_pose_15_v0` places eyes best once a bird is found, both on CUB and on the library.
2. **Small birds:** our full-frame pose models miss most small birds in the library. The external model's lead came from running on a crop of a detector box, and that box came from our own `bird_detect_v1`.
3. **Two-stage localization** (detector box → pose on the crop) is the measured win **over this repo's full-frame `localize_eyes()`**. With `bird_pose_15_v0`, library eyes within 5% of the bird-box diagonal go from 0.51 to 0.69, and for small birds from 0.32 to 0.68, with no change on CUB. It is available as an opt-in in PR #24.
4. **The fleet's eyes already come from a two-stage path.** The scoring pipeline's crop path (`modules/keypoints.py`: `bird_detect_v1` region, 25% padding, `eye_pose_v0`, IoU match, consumer floor 0.8) scores 0.67 within 5%. The two-stage `bird_pose_15_v0` is only +0.03 over it (paired 95% CI −0.04 to +0.09). For the fleet, the measured gap is detector false positives, not localization.
5. **Detector false positives are the fleet's measured gap, and a light second-detector check fixes most of them.** Requiring any RTMDet bird overlap (agreement ≥ 0.10) cut no-bird false eyes on all three labelled sets with no measured loss of eyes or birds; see the confirmation section. `bird_detect_v1` alone mistakes many other animals for birds.
6. **Detector false positives** are the main cost of two-stage. `bird_detect_v1` alone mistakes many other animals for birds; agreement with RTMDet is a strong filter.
7. **CUB numbers overstate library performance** by a wide margin: 0.96–0.98 versus 0.44–0.51 within 5% for the current full-frame models.

## Models compared

| ID | Model | How it runs | Size |
|---|---|---|---|
| A | `eye_pose_v0` (YOLO11n-pose, 6 points) | full frame, 640 px | 5.65 MB |
| A_pipe | the same through this repo's `PoseLocalizer.localize_eyes()` | full frame with its gates (what `eye-quality score` uses by default) | — |
| P_crop | the scoring pipeline's crop path, replayed (`pipeline_crop` adapter) | each `bird_detect_v1` region padded 25%, `eye_pose_v0` on the crop, pose detection chosen by IoU; consumers keep eyes ≥ 0.8. **This is what the fleet stores and shows.** | — |
| B | `bird_pose_15_v0` (YOLO11n-pose, 15 CUB parts) | full frame, 640 px | 5.97 MB |
| C_det | external model | crop of the `bird_detect_v1` box | — |
| C_gt | the same | labelled box (an "oracle" box: tests eye placement only) | — |
| A_td / B_td | A or B two-stage | `bird_detect_v1` box, 25% padding, pose on the crop | — |

Not compared: the external mammal-eye model (no bird keypoints) and the base `yolo11n-pose` (human keypoints).

## Data

### CUB held-out species

- **Split replay:** `eye_pose_v0`'s split was replayed from `training/convert_cub200.py`. `prepare_from_coco` shuffles the species folders with seed 42 and keeps 30 for validation.
- **No shared held-out species:** `bird_pose_15_v0` holds out a different 30 species, and **the two sets do not overlap**. Each model is therefore scored on its own held-out species (1,754 and 1,765 images); the other model has seen those species. No CUB subset is unseen by both.
- **External model:** its training data is unknown.

### Library sample (`data/field_eye_v1`, gitignored)

- **Sampling:** `training/sample_field_eyes.py` (read-only, seed `field-eye-v1`) drew 120 frames that the union of `bird_detect_v1` and RTMDet flagged as containing a bird:
  - 40 per size band, by the largest detector box's share of the frame: small < 4%, medium 4–15%, large ≥ 15%. Bands come from detector boxes, many of which turned out to be false (finding 10), so a band describes the detection, not necessarily a bird;
  - one frame per date folder;
  - camera-original folders only. By body: Z8 54, Z6ii 38, D90 18, phone/compact JPEG 10. D300 and DJI gave no flagged frames; for the D300 this is at least partly the RAW decoding failure in finding 13, which skipped many older-body NEFs.
- **Negatives:** 27 frames from `verified_negatives_v1` (three of the 30 drawn could not be decoded).
- **Effort and integrity:** 423 tries. All 147 source files were confirmed unmodified (size and mtime), and every rendition hash matched.
- **Renditions:** display-oriented JPEGs with a 2048 px long edge, shared by every model and by the labelling page.

### Owner labels

- **Labelling:** with `training/label_eyes/`, the owner labelled all 147 frames and excluded none. 152 subjects were boxed, 38 of them drawn by hand because neither detector had found them.
- **Non-bird subjects:** the owner boxed **every animal subject**, not only birds. A visual triage (`subjects.jsonl`) tagged 82 subjects as birds and 70 as other: cats, deer, squirrels, rabbits, goats, raccoon, opossum, lizards, turtles, a snake, a frog, insects and spiders, figurines and toys, and one person.
- **Bird-only labels:** eye slots are 78 visible, 57 hidden, 25 unresolvable and 4 unknown. 13 bird boxes were hand-drawn, and 9 frames hold more than one bird.
- **No closed eyes** were tagged, so eye-state outputs could not be checked.
- **Bird-only scores** use 84 frames: 63 with birds and 21 with no labelled subject. Frames whose subjects are all non-birds are left out.

## Method and checks

- **Scoring code:** `training/eval_eye_benchmark.py`.
  - Predicted eyes are matched to labelled eyes by minimum total distance, ignoring left/right.
  - Error is normalised by the labelled box diagonal.
  - PCK@t is the share of visible eyes within t; a missed bird counts its eyes as wrong.
- **Thresholds:** eye-confidence thresholds were frozen on CUB, using each model's own held-out set and Youden's J for visible versus hidden. **Nothing was tuned on the library sample.** "Ungated" scores count every predicted eye and measure placement alone.
- **Intervals:** 95% bootstrap over frames. Differences between two setups use a **paired** bootstrap over the same frames. No-subject rates rest on n = 21 frames.
- **Cross-check:** at confidence 0.5, `bird_pose_15_v0` scores 0.9575 within 5% on its held-out species, against 0.9558 from `training.eval_bird15_parts`. The small difference comes from left/right-free matching.
- **Production integration check:** the real `localize_eyes()` without two-stage reproduces the benchmark's production-path number exactly (0.4872).

## Results

### CUB (each model on its own held-out species, thresholded)

| Model | Within 2% | Within 5% [95% CI] | Within 10% | Median error | False eyes on hidden | Visible vs hidden AUROC |
|---|---|---|---|---|---|---|
| A | 0.699 | 0.976 [0.969–0.983] | 0.985 | 0.014 | 0.032 | 0.881 |
| A_pipe | 0.699 | 0.976 | 0.985 | 0.014 | 0.032 | 0.979 |
| B | 0.738 | 0.959 [0.949–0.968] | 0.969 | 0.013 | 0.032 | 0.943 |
| B_td | 0.730 | 0.960 [0.951–0.969] | 0.971 | 0.014 | 0.034 | 0.939 |
| C_det | 0.476 | 0.856 | 0.912 | 0.020 | 0.10–0.11 | 0.911 |
| C_gt | 0.479 | 0.870 | 0.939 | 0.020 | 0.09–0.12 | 0.91–0.92 |

### Library, birds only (thresholded)

| Setup | Birds found | Within 2% | Within 5% [95% CI] | Within 10% | Ungated within 5% | Frames with an eye found | False eyes on hidden | No-subject frames with a false eye |
|---|---|---|---|---|---|---|---|---|
| A | 0.60 | 0.22 | 0.44 [0.32–0.58] | 0.51 | 0.51 | 0.68 | 0.00 | 0.14 |
| A_pipe | 0.56 | 0.23 | 0.49 [0.36–0.62] | 0.58 | 0.49 | 0.70 | 0.21 | 0.14 |
| B | 0.62 | 0.32 | 0.51 [0.39–0.65] | 0.55 | 0.64 | 0.71 | 0.00 | 0.10 |
| C_det | 0.76 | 0.32 | 0.55 [0.43–0.69] | 0.67 | 0.67 | 0.79 | 0.12 | 0.29 |
| C_gt | 1.00 | 0.38 | 0.63 [0.51–0.74] | 0.81 | 0.76 | 0.91 | 0.16 | 0.00 |
| A_td | 0.76 | 0.41 | 0.69 [0.56–0.83] | 0.76 | 0.74 | 0.91 | 0.02 | 0.33 |
| **B_td** | 0.76 | **0.46** | **0.69 [0.57–0.82]** | 0.72 | **0.82** | 0.91 | 0.02 | 0.19 |
| P_crop, floor 0.8 (fleet) | 0.76 | 0.41 | 0.67 [0.54–0.80] | 0.73 | 0.74 | 0.91 | 0.02 | 0.19 |
| P_crop, gate 0.25 | 0.76 | 0.42 | 0.72 [0.58–0.86] | 0.79 | 0.74 | 0.91 | 0.09 | 0.33 |

**By bird size** (thresholded within 5%; share of birds found in brackets; small, medium and large hold 31, 31 and 16 visible eyes):

| Setup | Small | Medium | Large |
|---|---|---|---|
| A | 0.26 (0.27) | 0.48 (0.84) | 0.69 (0.93) |
| B | 0.32 (0.30) | 0.55 (0.84) | 0.81 (1.00) |
| C_det | 0.55 (0.59) | 0.45 (0.84) | 0.75 (1.00) |
| B_td | **0.68** (0.59) | **0.61** (0.84) | **0.88** (1.00) |
| P_crop, floor 0.8 | **0.68** (0.59) | 0.58 (0.84) | 0.81 (1.00) |

**Eye placement on the 44 birds every model found** (47 eyes, ungated), with detection taken out of the picture:

| Model | Within 2% | Within 5% [95% CI] | Within 10% | Median error |
|---|---|---|---|---|
| A | 0.47 | 0.83 [0.70–0.94] | 0.98 | 0.021 |
| **B** | **0.62** | **0.96 [0.90–1.00]** | **1.00** | **0.018** |
| C_det | 0.47 | 0.81 [0.70–0.91] | 0.89 | 0.024 |
| C_gt | 0.51 | 0.77 [0.64–0.89] | 0.94 | 0.020 |

**Production `localize_eyes()`**, `eye_pose_v0` with production gates:

| | Default | Two-stage |
|---|---|---|
| Within 5% | 0.487 | 0.667 [0.53–0.82] |
| Small birds | 0.35 | 0.71 |
| Frames with an eye found | 0.70 | 0.89 |
| No-subject frames with a false eye | 0.14 | 0.33 |
| Latency | 1× | about 2× |

Re-run on `main` after #30 (two-stage box selection fix): every row is unchanged (0.487 → 0.667 within 5%, small birds 0.35 → 0.71, no-subject false eyes 0.14 → 0.33).

The 2× is relative, from one harness that runs `predict()` and then `localize_eyes()` for both paths. The absolute per-image medians in the library table measure each adapter on its own: 215 ms for A_pipe, which runs inference twice inside `localize_eyes()`; 83 ms for B_td (one detector pass and one crop pose); 159 ms for P_crop.

**Paired differences** (same 84 frames, within 5%, 2,000 resamples):

| Comparison | Difference | 95% CI |
|---|---|---|
| B_td − C_det | +0.14 | 0.04 to 0.25 |
| B_td − P_crop at 0.8 | +0.03 | −0.04 to 0.09 |
| P_crop at 0.8 − A_pipe | +0.18 | 0.08 to 0.29 |

**All 152 labelled subjects:** the scores are much lower; for example, B scores 0.36 within 5%. Bird models are not meant for mammals, reptiles or insects, so these numbers describe behaviour on non-bird subjects, not bird-eye quality.

## Findings

1. **`bird_pose_15_v0` places eyes most precisely.**
   - Library birds found by every model: 0.96 within 5% and 0.62 within 2%, against 0.81 and 0.47 for the external model.
   - CUB within 2%: 0.70–0.74 for ours thresholded (0.74–0.80 ungated), about 0.48 for the external model.
2. **Missed small birds cause most library errors.** Run on the full frame at 640 px, our pose models find only 27–30% of small birds.
3. **Two-stage fixes most of it.** B_td small-bird eyes within 5% go from 0.32 to 0.68, with CUB unchanged. A_td improves too (0.44 to 0.69), so the gain does not depend on the 15-part model. The fleet's pipeline path is already two-stage (summary point 4), so this gain is over this repo's default `localize_eyes()`, not over the fleet.
4. **The external model's advantage was the design, not the model.** It received the same `bird_detect_v1` boxes as B_td. The unpaired intervals overlap (0.69 [0.57–0.82] vs 0.55 [0.43–0.69]), but on the same frames B_td is ahead by +0.14 (paired 95% CI 0.04 to 0.25).
5. **The models fail on different birds.**
   - External model: dark, backlit and night birds.
   - Ours: owls and other birds facing the camera.
   
   The biggest-disagreement frames are in the private report.
6. **The external model's errors are spread out, not shifted.** Its mean offset on CUB is under 0.005 of the diagonal, and removing it changes nothing; its errors are about 3× more spread out than ours.
7. **Crop convention:** the external model works better with the **stretched** box crop that the teacher pipeline already uses than with a centred square crop. On CUB, ungated within 2% drops from 0.497 to 0.376 with the square crop.
8. **Left/right naming:** the external model names a side for only about 20% of eyes, and is right 63–74% of the time when it does. Ours always name a side, and are right 99.6–99.7% of the time on confident eyes.
9. **Visibility separation is weak on library photos.**
   - Visible versus hidden AUROC: 0.88–0.98 on CUB for our models, but 0.58–0.81 on the library.
   - The production gates bring `eye_pose_v0` from 0.58 to 0.73, but they leave false eyes on 21% of hidden sides.
10. **Detector false positives** (the sampling detectors on the 120 flagged frames):
    - only 63 frames really held a bird: 50 held only other animals or objects, and 7 nothing;
    - when **both** `bird_detect_v1` and RTMDet fired, 54 of 62 frames held a bird;
    - when **only** `bird_detect_v1` fired, 48 of 54 did not;
    - at confidence 0.25, `bird_detect_v1` also put a box on about half of the 21 no-subject frames.
    
    Two-stage inherits these false boxes.
11. **CUB is not a proxy for library performance.** The gap is about 0.45 within 5% for the current full-frame models. CUB birds are large, centred and well lit.
12. **`verified_negatives_v1` checks out.** None of the 27 sampled frames contains a bird; the 13 with labelled subjects hold insects, lizards, a frog, figurines, a deer and a person.
13. **Older-body RAW decoding:** the shared preview extractor (`training/build_eye_dataset.py`) failed on 33 of 423 sampled NEFs (23 D90, 7 Z6ii, 1 D300), and on 43 more in the calibration draw. Root cause: with more than one SubIFD the tag value is an offset array, which was read as a single offset, and the fallback marker scan then accepted a false JPEG start. All 73 failing files decode after the fix (separate PR). Older bodies are under-sampled in both samples.
14. **The external model's auxiliary outputs (eye state, orientation) were not evaluated:** no test-sample eye was tagged closed, and the calibration sample has only 2.
15. **Requiring a second detector to agree removes bird-free false eyes.** It costs recall and speed and needs an eye-threshold rule that also penalises hidden sides; see the calibration section.
16. **Labels need a visual check before they are trusted.** The calibration labels were meant to be birds only, but 2 of 70 labelled subjects are not birds: a cat (`n0016`) and a person (`b0107`). An earlier version of this report read `n0016` as an error in `verified_negatives_v1`; it was not, and the manifest is correct for that frame. Calibration numbers here exclude both subjects.
17. **The fleet path is close to the best setup.** At the 0.8 consumer floor, P_crop matches B_td on false eyes (0.02 hidden, 0.19 no-subject) and is 0.03 behind on placement (not significant). Lowering the floor to the provider's 0.25 gate raises within 5% to 0.72, but hidden-side false eyes go 0.02 → 0.09 and no-subject false eyes 0.19 → 0.33. Re-fitting the floor belongs on the calibration sample and in the pipeline repository.

## Detector calibration (follow-up, same day)

**Calibration set (`data/field_eye_calib_v1`):**
- 108 frames that the detectors flagged as containing a bird, plus 26 verified no-bird frames.
- Seed `field-eye-calib-v1`, drawn with `--exclude-manifest`, so it shares no folder and no file with the test sample.
- The owner labelled birds, and a visual check of all 70 labelled subjects found 68 birds and 2 others, which are excluded (`subjects.jsonl`, `--birds-only`). That leaves 132 frames: 54 with birds and 78 with none.
  - 68 birds;
  - eye slots: 56 visible, 62 hidden, 18 unresolvable; 2 tagged closed;
  - only 54 of the 108 detector-flagged frames contain a bird.

**Method:** `eval_eye_benchmark run --calibrate`.
- **Inputs:** two-stage predictions from `topdown` with `agree=rtmdet`. Each `bird_detect_v1` box carries the best tiled-RTMDet bird score among overlapping boxes (IoU ≥ 0.3).
- **Grid:** detector confidence 0.25–0.75 × agreement 0.0–0.8, with the eye threshold re-chosen per cell (Youden).
- **Objective, fixed before looking:** PCK@0.05 minus the share of bird-free frames with a false eye.

**Result:** both two-stage models choose **detector confidence ≥ 0.60 and RTMDet agreement ≥ 0.50**. The optimum lies inside the grid; agreement 0.60 is worse.

| Calibration set, B_td | Within 5% | Birds found | No-bird frames with a false eye |
|---|---|---|---|
| Current default (conf 0.25, no agreement) | 0.73 | 0.76 | 0.71 |
| Calibrated (conf 0.60, agreement ≥ 0.50) | 0.71 | 0.66 | 0.06 |

**Frozen settings scored once on the test sample** (birds only):

| Test set, B_td | Birds found | Within 2% | Within 5% | Frames with an eye found | False eyes on hidden | No-subject frames with a false eye | ms/img |
|---|---|---|---|---|---|---|---|
| Before calibration | 0.76 | 0.46 | 0.69 | 0.91 | 0.02 | 0.19 | 83 |
| Calibrated | 0.63 | 0.49 | 0.69 | 0.80 | 0.19 | **0.00** | ~770 |

**What the calibration shows:**
- **The agreement rule works:** false eyes on bird-free frames went to zero on held-out data, with eye placement unchanged.
- **Detection cost:** about 12 points of bird recall.
- **Hidden-side cost:** the calibrated eye threshold is near zero (0.037; 0.057 for A_td), because the objective did not penalise false eyes on hidden sides. On the test set those rose from 0.02 to 0.19. The objective was not changed after seeing the test numbers. The next calibration should add a hidden-side term, decided in advance and re-run on fresh calibration data.
- **Speed cost:** tiled RTMDet makes two-stage about 10× slower (about 0.8 s per 2048 px image on the RTX 4060). A cheaper second check, such as a single untiled RTMDet pass on the box crop, should be tested before production use.
- **Cheaper check, measured on the calibration set only:** one untiled RTMDet pass per padded box crop (`agree=rtmdet_crop`) reaches the same best objective (0.650), with recall 0.68 vs 0.66, at a median 346 ms vs 635 ms per image. That is still about 4× the two-stage cost without a second detector.
- **Not yet in production:** the agreement rule exists only in the benchmark (`topdown,agree=rtmdet|rtmdet_crop`, `--agree-min`). `PoseLocalizer --two-stage` and the pipeline's crop path still use `bird_detect_v1` alone at confidence 0.25.

## Consumer floor sweep (calibration set only)

Follows #28. This uses the `pipeline_crop` predictions on the 134-frame calibration sample. The test sample was not used. Its no-bird frames (78) are hard, because many hold other animals.

**Objective, fixed before looking:** PCK@0.05 minus the false-eye rate on hidden sides minus the share of no-bird frames with a false eye, equally weighted.

| Setting (`eye_pose_v0`, two-stage) | Objective | Within 5% | False eyes on hidden | No-bird frames with a false eye | Frames with an eye found |
|---|---|---|---|---|---|
| Fleet today: floor 0.8 | 0.00 | 0.63 | 0.06 | 0.56 | 0.88 |
| Floor 0.97 | 0.22 | 0.59 | 0.06 | 0.31 | 0.83 |
| Floor 0.995 | 0.24 | 0.45 | 0.03 | 0.18 | 0.63 |
| **Floor 0.8 + agreement** (detector conf ≥ 0.60, tiled RTMDet agreement ≥ 0.50) | **0.48** | 0.55 | 0.06 | **0.01** | 0.78 |

**What the sweep shows:**
- **The floor is the wrong tool for detector false positives.** The pipeline's false boxes often hold real animals, so eye confidence on them stays high. Raising the floor keeps paying in real eyes, and the objective never settles: bootstrap picks split between 0.985 and 0.995. No change to the 0.8 floor is recommended on this evidence.
- **Detector agreement is the right tool.** With the 0.8 floor kept, it removes nearly all false eyes on no-bird frames and leaves hidden-side false eyes at 0.06. That also resolves the hidden-side cost in the calibration section, which came from a near-zero eye threshold, not from the agreement rule.
- **Cost:** about 7 points of eyes within 5% and 10 of frames with an eye found, from birds the agreement rule drops.
- **Check:** two-stage `eye_pose_v0` with no agreement at floor 0.8 reproduces the fleet path to within 0.01 (0.63, 0.06, 0.58).

This is calibration-set evidence with 56 visible eyes, and the combination was chosen after seeing the earlier test-set result, so confirm it on fresh labels before changing the fleet.

## Confirmation on fresh labels

**Confirmation set (`data/field_eye_confirm_v1`):**
- 70 detector-flagged frames plus 30 verified no-bird frames, sharing no folder or file with the two earlier samples. The target was 120 flagged frames; the unused folders ran out at one frame per folder.
- The owner labelled birds only. A visual check of all 50 labelled subjects found 43 birds and 7 others (lizard, turtles, dragonfly, frog, toy), which are excluded.
- Bird-only scoring uses 96 frames: 33 with birds (33 visible eyes, 31 hidden slots) and 63 with none. Only one visible eye is in the large-bird band.

**Pre-set test, scored once:** the fleet path at floor 0.8 against `eye_pose_v0` two-stage at floor 0.8 with the calibrated strict rule (detector confidence ≥ 0.60, tiled RTMDet agreement ≥ 0.50). Paired 95% intervals over frames.

| | Fleet, floor 0.8 | Strict rule | Difference [95% CI] |
|---|---|---|---|
| No-bird frames with a false eye | 0.43 | 0.02 | −0.41 [−0.54, −0.28] |
| False eyes on hidden sides | 0.16 | 0.06 | −0.10 [−0.20, 0.00] |
| Eyes within 5% | 0.58 | 0.48 | −0.09 [−0.22, 0.04] |
| Frames with an eye found | 0.83 | 0.67 | −0.17 [−0.33, −0.04] |
| Birds found | 0.72 | 0.44 | −0.28 [−0.42, −0.16] |

The strict rule's benefit is confirmed, but it drops far more birds than the calibration set suggested. **It should not ship.**

**Exploratory follow-up.** Separating the two thresholds on this set showed the cost comes from their strictness, not from requiring agreement. A **light rule**, detector confidence ≥ 0.25 (the current setting) plus RTMDet agreement ≥ 0.10, kept every bird. This rule was suggested by the confirmation set, so that row is not independent; the calibration and test sets played no part in suggesting it.

| Set (light rule vs fleet at floor 0.8) | No-bird frames with a false eye | Difference [95% CI] | Eyes within 5% | Birds found |
|---|---|---|---|---|
| Calibration (78 no-bird frames, 56 eyes) | 0.56 → 0.22 | −0.35 [−0.46, −0.24] | 0.63 → 0.63 | 0.76 → 0.76 |
| Test (21, 78 eyes) | 0.19 → 0.05 | −0.14 [−0.32, 0.00] | 0.67 → 0.67 | 0.76 → 0.76 |
| Confirmation (63, 33 eyes; suggested the rule) | 0.43 → 0.08 | −0.35 [−0.47, −0.24] | 0.58 → 0.61 | 0.72 → 0.72 |

**Recommendation:** the light rule is the fleet candidate: fewer false eyes with no measured loss of eyes or birds on any set. The small confirmation-set gain in eyes probably reflects a minor difference between the two adapters, not a real improvement.

**Cheaper check at the light setting.** The tiled RTMDet check took a median 421 ms per image against 69 ms for the fleet path, about 6×. One untiled RTMDet pass per box crop (`agree=rtmdet_crop`) was measured with the setting fixed in advance (detector ≥ 0.25, agreement ≥ 0.10, floor 0.8), on all three sets. It costs about 2× the fleet path (median about 200 ms against 105 ms, in the same runs):

| No-bird frames with a false eye | Fleet | Tiled check | Per-crop check | Per-crop − fleet [95% CI] |
|---|---|---|---|---|
| Calibration | 0.56 | 0.22 | 0.26 | −0.31 [−0.43, −0.20] |
| Test | 0.19 | 0.05 | 0.10 | −0.10 [−0.30, 0.09] |
| Confirmation | 0.43 | 0.08 | 0.14 | −0.29 [−0.40, −0.17] |

Eyes within 5% did not fall on any set. Birds found were unchanged on calibration and test; on confirmation one bird was lost (0.72 → 0.70). The per-crop check removes slightly fewer false eyes than the tiled one at about a third of the cost, so it is the practical choice. On the test set alone the improvement is not significant, with only 21 no-bird frames.

**Still open:**
- **False eyes that remain:** 0.05–0.26 of no-bird frames still get one, depending on the set and the check used.
- **Implementation:** the rule exists only in the benchmark; the change itself belongs in the pipeline repository.

## Pros and cons

**`bird_pose_15_v0`:**
- Pros: the best eye placement; few false eyes on hidden sides; reliable left/right; head, beak and body parts included.
- Cons: misses small birds on the full frame; weaker on owls and frontal heads; trained on CUB only.

**`eye_pose_v0` and this repo's default path:**
- Pros: close to B on CUB; the production gates improve visibility separation.
- Cons: behind B on library placement; 21% false eyes on hidden sides through the production gates; the slowest path, because `localize_eyes()` runs inference twice; the same small-bird problem.

**The pipeline's crop path (fleet):**
- Pros: already two-stage, so it handles small birds; the 0.8 floor keeps false eyes low on hidden sides.
- Cons: inherits `bird_detect_v1` false boxes (0.19 of no-subject frames get a false eye); uses `eye_pose_v0`, which places eyes slightly less precisely than `bird_pose_15_v0`.

**External model:**
- Pros: top-down, so it handles small birds given a box; small and fast; auxiliary outputs beyond keypoints.
- Cons: less precise placement; more false eyes on hidden sides (0.10–0.16); fails on dark birds; rarely names a side; depends on a detector box; unknown training data; training on its outputs is gated on permission.

## Improvements, ranked by measured gap

| # | Change | Evidence | Status |
|---|---|---|---|
| 1 | Two-stage localization in this repo's `localize_eyes()` (the fleet path already is two-stage) | Library within 5%: 0.49 → 0.67; small birds 0.35 → 0.71 | Opt-in in PR #24; off by default; box selection fixed in #30 |
| 2 | Calibrate the detector before two-stage becomes the default; test an RTMDet-agreement rule | Single-detector `bird_detect_v1` boxes were 89% false; no-subject false eyes 0.14 → 0.33 with two-stage | The strict rule (conf ≥ 0.60, agreement ≥ 0.50) was confirmed but drops too many birds. The light rule (conf ≥ 0.25, agreement ≥ 0.10) cuts false eyes on all three sets with no measured loss. Open: latency, pipeline implementation |
| 3 | Make `bird_pose_15_v0` the eye model (here and in the fleet crop path) | Best placement on both datasets; over the fleet path only +0.03 (paired CI −0.04 to 0.09) | Not now: keep `eye_pose_v0` until a significant gain over the fleet path is shown; it would also need a mapping from its parts (6 and 10) into the 6-point schema |
| 4 | Fine-tune on library-domain heads: owls, frontal heads, low light | Failure frames in the private report | Needs a separately labelled set; the 147-frame sample stays a test set |
| 5 | Learned visibility and eye state instead of thresholds | Library visible-vs-hidden AUROC 0.58–0.81 | Design work |
| 6 | Distillation from the external model, mainly for small-bird recall and visibility labels | Its strengths are small birds and frontal heads, not precision | Permission granted 2026-11-03; on hold under the T-series audit gates (human eye labels with left/right identity, partial-label trainer, frozen evaluation) |
| 7 | Fix decoding of older NEF previews | 76 sampled NEFs undecodable (33 test draw, 43 calibration draw) | Fixed in a separate PR; all 73 retested files decode |

## Limitations

- **Small sample:** 78 visible bird eyes, so intervals are about ±0.12 on library PCK@0.05.
- **Sampling favours detectable birds:** frames were drawn where a detector found something. Birds both detectors miss everywhere in a frame are not represented, so library recall is optimistic for detector-based setups. The 13 hand-drawn bird boxes partly offset this.
- **Bird / non-bird tags** come from a visual triage, not from owner labels.
- **No CUB subset is unseen by both of our models**, and the external model's training data is unknown.
- **The GPU test suite was not run.** The benchmark itself ran with real weights on CUDA.

## Reproduce

```powershell
python -m training.sample_field_eyes --root <library> --out data/field_eye_v1 --bodies "<camera folders>"
python -m training.label_eyes.serve data/field_eye_v1
python -m training.eval_eye_benchmark run --cub data/cub_bird15 --subset b15_val `
  --model "B=yolo,weights=models/bird_pose_15_v0.pt,eyes=6:10" --out data/eye_benchmark/cub_b15_val
python -m training.eval_eye_benchmark run --field data/field_eye_v1 --birds-only --thresholds <frozen.json> `
  --model "B=yolo,weights=models/bird_pose_15_v0.pt,eyes=6:10" `
  --model "B_td=topdown,weights=models/bird_pose_15_v0.pt,eyes=6:10" `
  --model "P_crop=pipeline_crop,weights=models/eye_pose_v0.pt,eyes=1:2" --out data/eye_benchmark/field
```

Data, labels, predictions and metrics stay in gitignored `data/`.
