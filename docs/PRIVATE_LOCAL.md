---
type: Guide
title: Local private documentation
description: Gitignored docs/private/ holds proprietary teacher manifests, hashes, and third-party reference inventories.
resource: docs/PRIVATE_LOCAL.md
tags: [guide, privacy, clean-room]
timestamp: 2026-09-29T00:00:00Z
okf_version: 0.1
---

# Local private documentation

Store proprietary content only under **`docs/private/`** or **`.docs/private/`** at the repo root.
Both directories are in `.gitignore` and must never be committed. Use them for:

- Proprietary teacher manifests (hashes, on-disk weight paths, training-use attestation)
- Third-party product names and external reference inventories

Public guides describe **open** teachers only (for example Apache-2.0 RTMDet-tiny). Proprietary pose-teacher
identity never appears in committed docs or default CLI paths.

After cloning, maintainers with licensed weights add the private manifest and lineage notes described in
your internal runbook (files live only under those private directories).

Verification: `python -m training.teacher.verify_teacher_provenance --gate gate0` checks the open detector
manifest always; proprietary pose checks run only when the private manifest is present.
