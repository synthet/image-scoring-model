#!/usr/bin/env python3
"""Convert CUB-200-2011 part annotations to YOLO pose dataset (6-keypoint bird schema)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from eye_quality.localization.keypoint_schema import KEYPOINT_NAMES, KPT_SHAPE
from training.prepare_dataset import prepare_from_coco

KPT_COUNT = KPT_SHAPE[0]
MIN_BBOX_AREA_FRAC = 0.002


def _load_parts(parts_file: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for line in parts_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        pid, name = line.split(maxsplit=1)
        mapping[int(pid)] = name.strip().lower().replace(" ", "_")
    return mapping


def _load_images(images_file: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for line in images_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        iid, rel = line.split(maxsplit=1)
        mapping[int(iid)] = rel.strip()
    return mapping


def _load_bboxes(bbox_file: Path) -> dict[int, tuple[float, float, float, float]]:
    mapping: dict[int, tuple[float, float, float, float]] = {}
    for line in bbox_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        iid, x, y, w, h = line.split()
        mapping[int(iid)] = (float(x), float(y), float(w), float(h))
    return mapping


def _load_part_locs(part_locs_file: Path) -> dict[int, dict[str, tuple[float, float, int]]]:
    """image_id -> part_name -> (x, y, visible)."""
    by_image: dict[int, dict[str, tuple[float, float, int]]] = defaultdict(dict)
    for line in part_locs_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        iid, pid, x, y, vis = line.split()
        # part_id resolved by caller
        by_image[int(iid)][int(pid)] = (float(x), float(y), int(vis))
    return by_image


def _part(
    parts: dict[int, tuple[float, float, int]],
    part_ids: dict[int, str],
    name: str,
) -> tuple[float, float, int] | None:
    for pid, pname in part_ids.items():
        if pname == name and pid in parts:
            return parts[pid]
    return None


def _normalize_name(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def _infer_bilateral_eyes(
    left_eye: tuple[float, float, int] | None,
    right_eye: tuple[float, float, int] | None,
    forehead: tuple[float, float, int] | None,
    beak: tuple[float, float, int] | None,
    head_top: tuple[float, float, int] | None,
    bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float, int], tuple[float, float, int]]:
    """Use CUB bilateral eyes when present; otherwise infer from head geometry."""
    if left_eye and left_eye[2] > 0 and right_eye and right_eye[2] > 0:
        return left_eye, right_eye
    if left_eye and left_eye[2] > 0:
        offset = max(bbox[2] * 0.12, 8.0)
        return left_eye, (left_eye[0] + offset, left_eye[1], 0)
    if right_eye and right_eye[2] > 0:
        offset = max(bbox[2] * 0.12, 8.0)
        return (right_eye[0] - offset, right_eye[1], 0), right_eye

    # Fallback: single visible head landmark
    anchor = forehead or head_top or beak
    if anchor and anchor[2] > 0:
        ax, ay, av = anchor
        offset = max(bbox[2] * 0.10, 8.0)
        return (ax - offset, ay, av), (ax + offset, ay, av)

    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h * 0.25
    offset = max(w * 0.10, 8.0)
    return (cx - offset, cy, 0), (cx + offset, cy, 0)


def _infer_shoulders(
    left_wing: tuple[float, float, int] | None,
    right_wing: tuple[float, float, int] | None,
    nape: tuple[float, float, int] | None,
    bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float, int], tuple[float, float, int]]:
    x, y, w, h = bbox
    cy = y + h * 0.45
    if left_wing and left_wing[2] > 0 and nape and nape[2] > 0:
        lx = nape[0] * 0.55 + left_wing[0] * 0.45
        ly = nape[1] * 0.55 + left_wing[1] * 0.45
        lv = min(left_wing[2], nape[2])
    elif left_wing and left_wing[2] > 0:
        lx, ly, lv = left_wing
    else:
        lx, ly, lv = x + w * 0.28, cy, 0

    if right_wing and right_wing[2] > 0 and nape and nape[2] > 0:
        rx = nape[0] * 0.55 + right_wing[0] * 0.45
        ry = nape[1] * 0.55 + right_wing[1] * 0.45
        rv = min(right_wing[2], nape[2])
    elif right_wing and right_wing[2] > 0:
        rx, ry, rv = right_wing
    else:
        rx, ry, rv = x + w * 0.72, cy, 0

    return (lx, ly, lv), (rx, ry, rv)


def map_cub_parts_to_keypoints(
    parts: dict[int, tuple[float, float, int]],
    part_ids: dict[int, str],
    bbox: tuple[float, float, float, float],
) -> list[float] | None:
    """Return flat COCO keypoints [x,y,v,...] for 6 schema points, or None to skip."""
    id_to_name = {pid: _normalize_name(n) for pid, n in part_ids.items()}

    def get(name: str) -> tuple[float, float, int] | None:
        for pid, pname in id_to_name.items():
            if pname == name and pid in parts:
                return parts[pid]
        return None

    beak = get("beak")
    left_eye = get("left_eye")
    right_eye = get("right_eye")
    crown = get("crown")
    forehead = get("forehead")
    left_wing = get("left_wing")
    right_wing = get("right_wing")
    nape = get("nape")

    head_top = crown if crown and crown[2] > 0 else forehead
    has_eye = (
        (left_eye and left_eye[2] > 0)
        or (right_eye and right_eye[2] > 0)
        or (forehead and forehead[2] > 0)
        or (crown and crown[2] > 0)
    )
    if not has_eye:
        return None

    le, re = _infer_bilateral_eyes(left_eye, right_eye, forehead, beak, head_top, bbox)
    left_sh, right_sh = _infer_shoulders(left_wing, right_wing, nape, bbox)

    named = {
        "beak": beak or (0.0, 0.0, 0),
        "left_eye": le,
        "right_eye": re,
        "head_top": head_top or forehead or (0.0, 0.0, 0),
        "left_shoulder": left_sh,
        "right_shoulder": right_sh,
    }

    flat: list[float] = []
    for kname in KEYPOINT_NAMES:
        px, py, pv = named[kname]
        flat.extend([float(px), float(py), int(pv)])
    return flat


def build_coco_json(
    cub_root: Path,
    *,
    min_bbox_area_frac: float = MIN_BBOX_AREA_FRAC,
) -> tuple[dict, dict[str, int]]:
    cub_root = Path(cub_root)
    part_ids = _load_parts(cub_root / "parts" / "parts.txt")
    images_map = _load_images(cub_root / "images.txt")
    bboxes = _load_bboxes(cub_root / "bounding_boxes.txt")
    raw_locs = _load_part_locs(cub_root / "parts" / "part_locs.txt")

    # Re-key part_locs by name
    part_locs: dict[int, dict[int, tuple[float, float, int]]] = defaultdict(dict)
    for iid, by_pid in raw_locs.items():
        for pid, loc in by_pid.items():
            part_locs[iid][pid] = loc

    stats = {"total": 0, "skipped_no_bbox": 0, "skipped_small_bbox": 0, "skipped_no_eye_head": 0, "kept": 0}
    coco_images: list[dict] = []
    coco_anns: list[dict] = []
    ann_id = 1

    from PIL import Image

    for iid, rel_path in sorted(images_map.items()):
        stats["total"] += 1
        bbox = bboxes.get(iid)
        if not bbox:
            stats["skipped_no_bbox"] += 1
            continue

        img_path = cub_root / "images" / rel_path.replace("/", "\\") if "\\" in rel_path else cub_root / "images" / rel_path
        if not img_path.is_file():
            # try forward slashes on Windows
            img_path = cub_root / "images" / Path(rel_path.replace("\\", "/"))
        if not img_path.is_file():
            stats["skipped_no_bbox"] += 1
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        x, y, w, h = bbox
        if (w * h) / (width * height) < min_bbox_area_frac:
            stats["skipped_small_bbox"] += 1
            continue

        kpts = map_cub_parts_to_keypoints(part_locs.get(iid, {}), part_ids, bbox)
        if kpts is None:
            stats["skipped_no_eye_head"] += 1
            continue

        file_name = rel_path.replace("\\", "/")
        coco_images.append(
            {"id": iid, "file_name": file_name, "width": width, "height": height}
        )
        coco_anns.append(
            {
                "id": ann_id,
                "image_id": iid,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "keypoints": kpts,
                "num_keypoints": sum(1 for i in range(2, len(kpts), 3) if kpts[i] > 0),
            }
        )
        ann_id += 1
        stats["kept"] += 1

    coco = {
        "info": {"description": "CUB-200-2011 converted to wildlife_bird 6-kpt schema"},
        "images": coco_images,
        "annotations": coco_anns,
        "categories": [{"id": 1, "name": "bird"}],
    }
    return coco, stats


def convert_cub200(
    cub_root: Path,
    output_dir: Path,
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
    coco_cache: Path | None = None,
) -> dict[str, int]:
    """Build COCO JSON from CUB and prepare YOLO pose dataset."""
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco, stats = build_coco_json(cub_root)
    cache_path = coco_cache or (output_dir / "cub200_coco.json")
    cache_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"Wrote COCO cache: {cache_path}")
    print(f"Stats: {stats}")

    images_dir = Path(cub_root) / "images"
    prepare_from_coco(cache_path, images_dir, output_dir, val_ratio)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CUB-200-2011 to YOLO pose dataset")
    parser.add_argument("--cub-root", required=True, help="Path to extracted CUB_200_2011 folder")
    parser.add_argument("--output", default="data/wildlife_bird", help="Output YOLO dataset dir")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_cub200(
        Path(args.cub_root),
        Path(args.output),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
