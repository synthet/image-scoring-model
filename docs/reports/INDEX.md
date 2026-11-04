---
type: Documentation Index
title: Reports index
description: Dated measurement and experiment reports.
resource: docs/reports/INDEX.md
tags: [docs, index, reports]
timestamp: 2026-09-25T00:00:00Z
okf_version: 0.1
---

# Reports

Dated snapshots. Numbers are measurements on our own data; thresholds are starting points to re-fit.

| Report | Summary |
|--------|---------|
| [eye-model-benchmark-2026-11-03.md](eye-model-benchmark-2026-11-03.md) | Our YOLO eye models, the pipeline crop path and an external top-down model on CUB and 147 owner-labelled library frames: label statistics, 17 findings, detector calibration, pros and cons, ranked improvements. The fleet path already reaches 0.67 PCK@0.05; a light second-detector check cuts its false eyes with no measured loss |
| [teacher-pseudo-labels-v0-2026-09-25.md](teacher-pseudo-labels-v0-2026-09-25.md) | Upstream RTMDet-tiny as a teacher: 877 pseudo-labelled images (503 bird boxes, 419 hard negatives) for `bird_detect_v0`, policy calibrated on the #377 owner labels |
