"""Verify on-disk teacher weights match recorded Gate 0 manifests.

    python -m training.teacher.verify_teacher_provenance --gate gate0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from training.teacher.proprietary_teacher_paths import proprietary_pose_manifest_path

REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_file(rel: str, expected: str, label: str) -> list[str]:
    path = REPO / rel
    errors: list[str] = []
    if not path.is_file():
        errors.append(f"missing {label}: {rel}")
        return errors
    actual = sha256(path)
    if actual != expected.lower():
        errors.append(f"hash mismatch {label} {rel}: got {actual}, want {expected}")
    return errors


def verify_rtmdet() -> list[str]:
    manifest_path = REPO / "models" / "rtmdet_tiny_coco.manifest.json"
    if not manifest_path.is_file():
        return ["missing models/rtmdet_tiny_coco.manifest.json"]
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    errors.extend(_check_file("models/rtmdet_tiny_coco.pth", m["checkpoint_sha256"], "RTMDet ckpt"))
    errors.extend(_check_file("models/rtmdet_tiny_coco.onnx", m["onnx_sha256"], "RTMDet onnx"))
    mod = REPO / m["module"]
    if mod.is_file():
        if sha256(mod) != m["module_sha256"]:
            errors.append(f"hash mismatch RTMDet module {m['module']}")
    else:
        errors.append(f"missing RTMDet module {m['module']}")
    return errors


def verify_bird_eye() -> list[str]:
    manifest_path = proprietary_pose_manifest_path()
    if not manifest_path.is_file():
        return []
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for name, art in m.get("artifacts", {}).items():
        errors.extend(_check_file(art["path"], art["sha256"], f"bird-eye {name}"))
    adapter = m.get("adapter", {})
    for key, rel_key in (("module", "module_sha256"), ("alignment_module", "alignment_module_sha256")):
        rel = adapter.get(key)
        exp = adapter.get(rel_key)
        if rel and exp:
            path = REPO / rel
            if not path.is_file():
                errors.append(f"missing adapter {rel}")
            elif sha256(path) != exp:
                errors.append(f"hash mismatch adapter {rel}")
    perm = m.get("training_use_permission", {})
    if perm.get("status") != "granted":
        errors.append(
            "bird-eye training_use_permission.status is not 'granted' "
            f"(current: {perm.get('status')!r}); pseudo-label training blocked"
        )
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=("gate0",), default="gate0")
    ap.add_argument(
        "--allow-pending-permission",
        action="store_true",
        help="Only verify file hashes; do not require granted training permission",
    )
    args = ap.parse_args()
    if args.gate != "gate0":
        raise SystemExit(f"unsupported gate: {args.gate}")

    errors = verify_rtmdet()
    eye_errors = verify_bird_eye()
    if args.allow_pending_permission:
        eye_errors = [e for e in eye_errors if "training_use_permission" not in e]
    errors.extend(eye_errors)

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        raise SystemExit(1)
    print("Gate 0 teacher provenance OK")


if __name__ == "__main__":
    main()
