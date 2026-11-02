"""Import owner-reviewed bird-free source photos as a private training manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

_WSL_PHOTO = re.compile(r"^/mnt/[a-z]/Photos/(.+)$", re.IGNORECASE)
_IMAGE_SUFFIXES = {".nef", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".arw", ".cr2", ".cr3", ".dng", ".raf"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_photo_path(wsl_path: str, photo_root: Path, *, must_exist: bool = True) -> Path:
    """Map `/mnt/<drive>/Photos/...` below a chosen local Photos root."""
    match = _WSL_PHOTO.fullmatch(wsl_path.strip()) if isinstance(wsl_path, str) else None
    if not match:
        raise ValueError(f"photo path is not in /mnt/<drive>/Photos/: {wsl_path!r}")
    parts = match.group(1).split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("photo path escapes photo root or has empty components")
    root = photo_root.resolve()
    candidate = root.joinpath(*parts).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("photo path escapes photo root")
    if candidate.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ValueError(f"unsupported photo suffix: {candidate.suffix}")
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must have columns {sorted(required)}")
        return list(reader)


def _group(path: Path, root: Path) -> str:
    return path.parent.relative_to(root.resolve()).as_posix()


def build_manifest(
    cohort_csv: Path, labels_csv: Path, photo_root: Path, *, holdout_cohort: Path | None = None,
) -> list[dict]:
    """Keep only human `no_bird` labels; prohibit any holdout-folder overlap."""
    cohort_rows = _read_csv(cohort_csv, {"image_id", "file_path", "stratum"})
    label_rows = _read_csv(labels_csv, {"image_id", "label"})
    by_image: dict[str, dict[str, str]] = {}
    for row in cohort_rows:
        image_id = row["image_id"].strip()
        if not image_id or image_id in by_image:
            raise ValueError(f"missing or duplicate cohort image_id: {image_id}")
        by_image[image_id] = row
    labels: dict[str, str] = {}
    for row in label_rows:
        image_id = row["image_id"].strip()
        if image_id in labels:
            raise ValueError(f"duplicate owner label for image_id: {image_id}")
        label = row["label"].strip()
        if label not in {"bird", "no_bird", "unsure"}:
            raise ValueError(f"unrecognized owner label for {image_id}: {label}")
        labels[image_id] = label
    if set(labels) != set(by_image):
        raise ValueError("cohort and owner labels have different image IDs")

    holdout_groups: set[str] = set()
    if holdout_cohort is not None:
        for row in _read_csv(holdout_cohort, {"image_id", "file_path"}):
            path = resolve_photo_path(row["file_path"], photo_root, must_exist=False)
            holdout_groups.add(_group(path, photo_root))

    records: list[dict] = []
    for image_id in sorted(by_image, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
        if labels[image_id] != "no_bird":
            continue
        row = by_image[image_id]
        source = resolve_photo_path(row["file_path"], photo_root)
        group = _group(source, photo_root)
        if group in holdout_groups:
            raise ValueError(f"training group {group} overlaps holdout cohort")
        records.append({
            "schema_version": 1,
            "image_id": image_id,
            "source_image_path": str(source),
            "source_sha256": file_sha256(source),
            "source_size_bytes": source.stat().st_size,
            "group_id": group,
            "label": "human_verified_no_bird",
            "split": "train",
            "source_stratum": row["stratum"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--photo-root", type=Path, required=True)
    parser.add_argument("--holdout-cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to((root / "data").resolve()) or output.suffix != ".jsonl":
        parser.error("output must be a .jsonl file under this repo's gitignored data/")
    summary_path = output.with_suffix(".summary.json")
    if output.exists() or summary_path.exists():
        parser.error("output already exists; use a new versioned directory")
    records = build_manifest(args.cohort, args.labels, args.photo_root,
                             holdout_cohort=args.holdout_cohort)
    if not records:
        parser.error("no verified negatives found")
    summary = {
        "schema_version": 1,
        "record_count": len(records),
        "group_count": len({row["group_id"] for row in records}),
        "cohort_csv_sha256": file_sha256(args.cohort),
        "labels_csv_sha256": file_sha256(args.labels),
        "holdout_csv_sha256": file_sha256(args.holdout_cohort),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        staged.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
                          encoding="utf-8")
        staged.replace(output)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        staged.unlink(missing_ok=True)
    print(json.dumps({"manifest": str(output), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
