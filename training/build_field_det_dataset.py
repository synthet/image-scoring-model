#!/usr/bin/env python3
"""Build a detector fine-tune dataset from owner-labelled field frames, verified negatives and a CUB replay.

Field frames are listed --repeat times so the small field set keeps real weight next to the CUB replay. A frame
with no boxes is a hard negative. Any frame sharing a shooting date with a held-out set stops the build: folders
are split by lens, so two folders from one day can hold the same bird. Source photos are only read.

    python -m training.build_field_det_dataset --out data/det_field_d1 \
        --field data/field_det_train_v1 --field data/field_det_train_v1b --field data/field_eye_calib_v1 \
        --split data/field_det_train_v1/split.json --drop field_det_train_v1b:b0008 \
        --held-manifest data/field_eye_v1/manifest.jsonl --held-manifest data/field_eye_confirm_v1/manifest.jsonl \
        --held-cohort <promotion cohort.csv>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def shooting_date(path: str) -> str | None:
    """The last folder in `path` that is a YYYY-MM-DD date."""
    found = [p for p in str(path).replace("\\", "/").split("/")[:-1] if DATE_RE.fullmatch(p)]
    return found[-1] if found else None


def yolo_line(bbox: list[float], width: int, height: int) -> str:
    """Class 0 YOLO row from a pixel x1y1x2y2 box, clipped to the frame."""
    x1, x2 = (min(max(v, 0.0), width) for v in (bbox[0], bbox[2]))
    y1, y2 = (min(max(v, 0.0), height) for v in (bbox[1], bbox[3]))
    return f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def field_rows(sample: Path, allow: set[str] | None, drop: set[str]) -> list[dict]:
    """Reviewed frames of one sample, without boxes tagged as non-birds (those frames become hard negatives)."""
    manifest = {r["image_id"]: r for r in _jsonl(sample / "manifest.jsonl")}
    subjects = sample / "subjects.jsonl"
    other = {(r["image_id"], r["bird_index"]) for r in _jsonl(subjects) if r["subject"] != "bird"} if subjects.is_file() else set()
    out = []
    for label in _jsonl(sample / "labels.jsonl"):
        image_id = label["image_id"]
        if label.get("excluded") or image_id in drop or (allow is not None and image_id not in allow):
            continue
        m = manifest[image_id]
        boxes = [b["bbox"] for i, b in enumerate(label.get("birds", [])) if (image_id, i) not in other]
        out.append({"id": f"{sample.name}_{image_id}", "image_path": m["image_path"], "source_path": m["source_path"],
                    "width": m["width"], "height": m["height"], "boxes": boxes})
    return out


def held_dates(manifests: list[str], cohorts: list[str]) -> set[str]:
    dates = {shooting_date(r["source_path"]) for m in manifests for r in _jsonl(Path(m))}
    for cohort in cohorts:
        with open(cohort, encoding="utf-8", newline="") as fh:
            dates |= {shooting_date(r["file_path"]) for r in csv.DictReader(fh)}
    return {d for d in dates if d}


def check_held_out(sources: list[str], held: set[str]) -> None:
    clash = sorted(s for s in sources if shooting_date(s) in held or shooting_date(s) is None)
    if clash:
        raise SystemExit(f"{len(clash)} frames share a date with a held-out set or have no date, e.g. {clash[0]}")


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _cub_replay(cub: Path, split: str, count: int, out: Path, rng: random.Random) -> list[str]:
    """Link a seeded CUB sample and keep only the box columns of its pose labels."""
    images = sorted((cub / "images" / split).iterdir())
    listed = []
    for image in rng.sample(images, min(count, len(images))):
        rows = (cub / "labels" / split / f"{image.stem}.txt").read_text(encoding="utf-8").splitlines()
        dst = out / "images" / f"cub_{split}" / image.name
        _link(image, dst)
        _write_label(out / "labels" / f"cub_{split}" / f"{image.stem}.txt", [" ".join(r.split()[:5]) for r in rows if r.strip()])
        listed.append(dst.resolve().as_posix())
    return listed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", action="append", required=True, help="Labelled sample folder")
    ap.add_argument("--split", required=True, help="split.json naming the calibration and negative frames allowed in training")
    ap.add_argument("--calibration-sample", default="field_eye_calib_v1", help="Sample limited by the split record")
    ap.add_argument("--negatives-manifest", default="data/verified_negatives_v1/manifest.jsonl")
    ap.add_argument("--drop", action="append", default=[], help="sample:image_id to leave out, e.g. an incompletely boxed frame")
    ap.add_argument("--held-manifest", action="append", default=[])
    ap.add_argument("--held-cohort", action="append", default=[])
    ap.add_argument("--cub", default="data/cub_bird15")
    ap.add_argument("--cub-train", type=int, default=2000)
    ap.add_argument("--cub-val", type=int, default=500)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--seed", default="det-field-d1")
    ap.add_argument("--long-edge", type=int, default=2048)
    a = ap.parse_args()

    from training.sample_field_eyes import render

    out = Path(a.out)
    if (out / "data.yaml").exists():
        raise SystemExit(f"{out} is already built; choose a new --out")
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    rows = []
    for name in a.field:
        sample = Path(name)
        allow = set(split["train_from_calibration"]) if sample.name == a.calibration_sample else None
        rows += field_rows(sample, allow, {d.split(":", 1)[1] for d in a.drop if d.split(":", 1)[0] == sample.name})
    wanted = set(split["verified_negatives_date_clean"])
    negatives = [r for r in _jsonl(Path(a.negatives_manifest)) if r["image_id"] in wanted]
    if len(negatives) != len(wanted):
        raise SystemExit(f"{len(wanted) - len(negatives)} split negatives are missing from {a.negatives_manifest}")
    check_held_out([r["source_path"] for r in rows] + [r["source_image_path"] for r in negatives],
                   held_dates(a.held_manifest, a.held_cohort))

    field = []
    for r in rows:
        dst = out / "images" / "field" / f"{r['id']}.jpg"
        _link(Path(r["image_path"]), dst)
        _write_label(out / "labels" / "field" / f"{r['id']}.txt", [yolo_line(b, r["width"], r["height"]) for b in r["boxes"]])
        field.append(dst.resolve().as_posix())
    for r in negatives:
        name = f"verified_negative_{r['image_id']}"
        dst = out / "images" / "field" / f"{name}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        render(Path(r["source_image_path"]), a.long_edge).save(dst, format="JPEG", quality=93)
        _write_label(out / "labels" / "field" / f"{name}.txt", [])
        field.append(dst.resolve().as_posix())

    rng = random.Random(a.seed)
    train = _cub_replay(Path(a.cub), "train", a.cub_train, out, rng) + field * a.repeat
    val = _cub_replay(Path(a.cub), "val", a.cub_val, out, rng)
    rng.shuffle(train)
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: bird\n", encoding="utf-8")
    summary = {
        "seed": a.seed, "repeat": a.repeat, "cub_train": a.cub_train, "cub_val": a.cub_val, "drop": a.drop,
        "field_frames": len(rows), "field_boxes": sum(len(r["boxes"]) for r in rows),
        "field_no_bird": sum(not r["boxes"] for r in rows), "verified_negatives": len(negatives),
        "field_share_of_train": round(len(field) * a.repeat / len(train), 3),
        "train_listed": len(train), "val_listed": len(val),
        "split_sha256": hashlib.sha256(Path(a.split).read_bytes()).hexdigest(),
        "approved_by": split.get("approved_by"), "held_manifests": a.held_manifest, "held_cohorts": len(a.held_cohort),
    }
    (out / "build.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
