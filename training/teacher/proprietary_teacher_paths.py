"""Resolve gitignored proprietary teacher manifests (local maintainer machines only)."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_PROPRIETARY_POSE_MANIFEST_NAME = "gate0-proprietary-pose-teacher.manifest.json"
_PRIVATE_DOC_ROOTS = (
    REPO / "docs" / "private",
    REPO / ".docs" / "private",
)


def proprietary_private_roots() -> tuple[Path, ...]:
    """Gitignored directories for proprietary manifests and lineage notes."""
    return _PRIVATE_DOC_ROOTS


def proprietary_pose_manifest_path() -> Path:
    override = os.environ.get("GATE0_PROPRIETARY_POSE_MANIFEST")
    if override:
        return Path(override)
    for root in _PRIVATE_DOC_ROOTS:
        candidate = root / _PROPRIETARY_POSE_MANIFEST_NAME
        if candidate.is_file():
            return candidate
    return _PRIVATE_DOC_ROOTS[0] / _PROPRIETARY_POSE_MANIFEST_NAME
