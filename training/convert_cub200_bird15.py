#!/usr/bin/env python3
"""Convert CUB-200-2011 to a YOLO pose dataset with all 15 CUB parts (bird_pose_15).

Unlike `convert_cub200.py` (6 head keypoints for eye_pose_v0), this keeps every CUB part, including both wings,
both legs and the tail, so body pose (wing position, tail angle) can be measured. Keypoint order follows CUB
`parts/parts.txt` (ids 1..15). Images are hard-linked into the dataset (copied when linking fails); the split is a
seeded per-species split so no species leaks between train and val.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from training.convert_cub200 import _load_bboxes, _load_images, _load_parts

# CUB part ids 1..15 -> keypoint index 0..14
BIRD15_NAMES: tuple[str, ...] = (
    "back", "beak", "belly", "breast", "crown", "forehead", "left_eye", "left_leg", "left_wing", "nape",
    "right_eye", "right_leg", "right_wing", "tail", "throat",
)
# Horizontal flip swaps left/right eye, leg and wing.
BIRD15_FLIP_IDX: list[int] = [0, 1, 2, 3, 4, 5, 10, 11, 12, 9, 6, 7, 8, 13, 14]
MIN_BBOX_AREA_FRAC = 0.002
MIN_VISIBLE = 4


def _part_locs(path: Path) -> dict[int, dict[int, tuple[float, float, int]]]:
    out: dict[int, dict[int, tuple[float, float, int]]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            iid, pid, x, y, vis = line.split()
            out[int(iid)][int(pid)] = (float(x), float(y), int(float(vis)))
    return out


def yolo_line(bbox: tuple[float, float, float, float], parts: dict[int, tuple[float, float, int]],
              width: int, height: int) -> str:
    """`0 cx cy w h` then 15 x `x y v`, normalised; v=2 observed, 0 missing (x=y=0)."""
    x, y, w, h = bbox
    vals = [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]
    for pid in range(1, len(BIRD15_NAMES) + 1):
        px, py, pv = parts.get(pid, (0.0, 0.0, 0))
        vals += [px / width, py / height, 2] if pv > 0 else [0.0, 0.0, 0]
    return "0 " + " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals)


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert(cub_root: Path, out: Path, val_ratio: float = 0.15, seed: int = 42) -> dict[str, int]:
    from PIL import Image

    parts_names = _load_parts(cub_root / "parts" / "parts.txt")
    expected = {i + 1: n for i, n in enumerate(BIRD15_NAMES)}
    if parts_names != expected:
        raise SystemExit(f"Unexpected CUB parts list: {parts_names}")
    images = _load_images(cub_root / "images.txt")
    boxes = _load_bboxes(cub_root / "bounding_boxes.txt")
    locs = _part_locs(cub_root / "parts" / "part_locs.txt")
    species = sorted({rel.split("/")[0] for rel in images.values()})
    rng = random.Random(seed)
    val_species = set(rng.sample(species, max(1, round(len(species) * val_ratio))))
    stats = {"total": 0, "kept": 0, "skipped_small_or_missing": 0, "skipped_few_parts": 0, "train": 0, "val": 0}
    manifest = []
    for iid, rel in sorted(images.items()):
        stats["total"] += 1
        src = cub_root / "images" / rel
        bbox = boxes.get(iid)
        if bbox is None or not src.is_file():
            stats["skipped_small_or_missing"] += 1
            continue
        with Image.open(src) as im:
            width, height = im.size
        if bbox[2] * bbox[3] / (width * height) < MIN_BBOX_AREA_FRAC:
            stats["skipped_small_or_missing"] += 1
            continue
        parts = locs.get(iid, {})
        if sum(1 for v in parts.values() if v[2] > 0) < MIN_VISIBLE:
            stats["skipped_few_parts"] += 1
            continue
        split = "val" if rel.split("/")[0] in val_species else "train"
        name = rel.replace("/", "__")
        _place(src, out / "images" / split / name)
        label = out / "labels" / split / (Path(name).stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(yolo_line(bbox, parts, width, height) + "\n", encoding="utf-8")
        manifest.append({"image_id": iid, "file": rel, "split": split})
        stats["kept"] += 1
        stats[split] += 1
    (out / "manifest.json").write_text(json.dumps({"stats": stats, "val_species": sorted(val_species),
                                                   "keypoints": list(BIRD15_NAMES), "flip_idx": BIRD15_FLIP_IDX,
                                                   "images": manifest}, indent=1), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cub-root", required=True)
    ap.add_argument("--output", default="data/cub_bird15")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(convert(Path(a.cub_root), Path(a.output), a.val_ratio, a.seed)))


if __name__ == "__main__":
    main()
