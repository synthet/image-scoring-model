#!/usr/bin/env python3
"""Convert COCO keypoint JSON or CSV annotations to YOLO pose label format."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from eye_quality.localization.keypoint_schema import FLIP_IDX, KEYPOINT_NAMES, KPT_SHAPE

# Re-export for dataset YAML generation
KPT_COUNT = KPT_SHAPE[0]


def _coco_to_yolo_line(
    bbox_xywh: list[float],
    keypoints: list[float],
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> str:
    """COCO bbox + flat keypoints [x,y,v,...] -> YOLO pose label line."""
    x, y, w, h = bbox_xywh
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    parts = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
    for i in range(KPT_COUNT):
        ki = i * 3
        if ki + 2 < len(keypoints):
            kx = keypoints[ki] / img_w
            ky = keypoints[ki + 1] / img_h
            kv = int(keypoints[ki + 2])
        else:
            kx, ky, kv = 0.0, 0.0, 0
        parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
    return " ".join(parts)


def _load_coco(coco_path: Path) -> tuple[dict, dict]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)
    return images, anns_by_image


def _load_csv(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _group_key(row: dict, image_field: str) -> str:
    path = row.get(image_field, "")
    parent = str(Path(path).parent)
    return parent or "_root"


def prepare_from_coco(coco_path: Path, images_dir: Path, output_dir: Path, val_ratio: float) -> None:
    images_meta, anns_by_image = _load_coco(coco_path)
    groups: dict[str, list[int]] = defaultdict(list)
    for img_id, meta in images_meta.items():
        file_name = meta.get("file_name", "")
        src = images_dir / file_name
        if not src.is_file():
            continue
        groups[_group_key({"path": file_name}, "path")].append(img_id)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    n_val = max(1, int(len(group_keys) * val_ratio)) if group_keys else 0
    val_groups = set(group_keys[:n_val])

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for img_id, meta in images_meta.items():
        file_name = meta.get("file_name", "")
        src = images_dir / file_name
        if not src.is_file():
            continue
        group = _group_key({"path": file_name}, "path")
        split = "val" if group in val_groups else "train"
        safe_name = str(Path(file_name)).replace("\\", "__").replace("/", "__")
        dst_img = output_dir / "images" / split / safe_name
        shutil.copy2(src, dst_img)

        img_w = meta.get("width", 0)
        img_h = meta.get("height", 0)
        label_lines: list[str] = []
        for ann in anns_by_image.get(img_id, []):
            bbox = ann.get("bbox")
            kpts = ann.get("keypoints", [])
            if bbox and img_w and img_h:
                label_lines.append(_coco_to_yolo_line(bbox, kpts, img_w, img_h))

        label_path = output_dir / "labels" / split / f"{Path(safe_name).stem}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        manifest.append({"image": str(dst_img), "label": str(label_path), "split": split})

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} images under {output_dir}")


def prepare_from_csv(csv_path: Path, output_dir: Path, val_ratio: float) -> None:
    rows = _load_csv(csv_path)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, "image_path")].append(row)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    n_val = max(1, int(len(group_keys) * val_ratio)) if group_keys else 0
    val_groups = set(group_keys[:n_val])

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for row in rows:
        src = Path(row["image_path"])
        if not src.is_file():
            continue
        split = "val" if _group_key(row, "image_path") in val_groups else "train"
        dst_img = output_dir / "images" / split / src.name
        shutil.copy2(src, dst_img)

        img_w = int(row.get("width", 0))
        img_h = int(row.get("height", 0))
        bbox = json.loads(row.get("bbox_json", "[]"))
        kpts = json.loads(row.get("keypoints_json", "[]"))
        if not img_w or not img_h:
            from PIL import Image

            with Image.open(src) as im:
                img_w, img_h = im.size

        line = _coco_to_yolo_line(bbox, kpts, img_w, img_h)
        label_path = output_dir / "labels" / split / f"{src.stem}.txt"
        label_path.write_text(line + "\n", encoding="utf-8")
        manifest.append({"image": str(dst_img), "label": str(label_path), "split": split})

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} images under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO pose dataset for wildlife birds")
    parser.add_argument("--input", required=True, help="COCO JSON or CSV annotations")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--images-dir", default=None, help="Image root for COCO file_name paths")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.suffix.lower() == ".json":
        if not args.images_dir:
            raise SystemExit("--images-dir required for COCO JSON input")
        prepare_from_coco(input_path, Path(args.images_dir), output_dir, args.val_ratio)
    else:
        prepare_from_csv(input_path, output_dir, args.val_ratio)


if __name__ == "__main__":
    main()
