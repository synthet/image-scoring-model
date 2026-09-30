---
type: Report
title: Gate 0 — teacher provenance and training-use permission (T-series eye pose)
description: Open detector teacher identity for T-series eye-pose pseudo-labeling; proprietary pose teacher recorded only under docs/private/.
resource: docs/reports/teacher-gate0-t-series-2026-09-28.md
tags: [report, training, teacher-student, provenance, gate0, eye-pose]
timestamp: 2026-09-28T00:00:00Z
okf_version: 0.1
status: complete
---

# Gate 0 — teacher provenance and training-use permission (T-series)

This report satisfies **Gate 0** for the **open** detector teacher used before optional proprietary pose
pseudo-labeling. Plans under `docs/private/` cover distillation execution; they are not committed.

**Status:** `complete` for the Apache-2.0 detector teacher. Proprietary pose-teacher hashes and permission
are maintained locally ([PRIVATE_LOCAL.md](../PRIVATE_LOCAL.md)).

## Detector teacher (RTMDet-tiny COCO)

Used for bird boxes before an optional proprietary eye teacher crop.

| Item | Value |
|------|--------|
| Model | RTMDet-tiny, COCO 80 classes (bird = 14) |
| Licence | Apache-2.0 (OpenMMLab) |
| Pseudo-label training | **Permitted** per [TEACHER_PSEUDO_LABELS.md](../guides/TEACHER_PSEUDO_LABELS.md) |
| Manifest | `models/rtmdet_tiny_coco.manifest.json` (checkpoint, ONNX, and module hashes) |
| Runtime code | `training/teacher/rtmdet_tiny.py` |
| Pipeline | `training/teacher/pseudo_label_rtmdet.py` |

Verify locally:

```bash
python -m training.teacher.verify_teacher_provenance --gate gate0
```

## Proprietary pose teacher (local only)

Optional bird head/eye SimCC teacher weights are **not** named, hashed, or path-listed in this repository.
Maintainers store manifests, attestation, and third-party lineage under gitignored `docs/private/`.

When the private manifest is present, `verify_teacher_provenance` also checks proprietary artifact hashes
and `training_use_permission.status`.

## Gate checklist

| Step | Done |
|------|------|
| Record open det teacher in committed manifest | Yes |
| Document proprietary pose teacher off git | Yes |
| `verify_teacher_provenance` passes (open det; pose when private manifest exists) | `python -m training.teacher.verify_teacher_provenance --gate gate0` |

## References

- [teacher-pseudo-labels-v0-2026-09-25.md](./teacher-pseudo-labels-v0-2026-09-25.md) — RTMDet v0 pseudo-label run
- Backend upstream identity (public RTMDet only): image-scoring-backend `docs/reports/upstream-weights-identity-2026-09-25.md`
