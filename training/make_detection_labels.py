#!/usr/bin/env python3
"""Derive YOLO detect labels (box only) from an existing pose dataset."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _strip_pose_line(line: str) -> str:
    """Keep only <class> <cx> <cy> <w> <h> from a YOLO pose label line."""
    parts = line.strip().split()
    if len(parts) < 5:
        return ""
    return " ".join(parts[:5])


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "skipped"
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def make_detection_labels(
    pose_dir: Path,
    output_dir: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, int]:
    """Strip keypoints from pose labels and mirror images into a detect dataset."""
    pose_dir = Path(pose_dir)
    output_dir = Path(output_dir)
    counts: dict[str, int] = {}

    for split in splits:
        pose_labels = pose_dir / "labels" / split
        pose_images = pose_dir / "images" / split
        out_labels = output_dir / "labels" / split
        out_images = output_dir / "images" / split
        out_labels.mkdir(parents=True, exist_ok=True)
        out_images.mkdir(parents=True, exist_ok=True)

        label_files = sorted(pose_labels.glob("*.txt"))
        counts[f"labels_{split}"] = 0
        counts[f"images_{split}"] = 0

        for label_path in label_files:
            lines = label_path.read_text(encoding="utf-8").splitlines()
            det_lines = [_strip_pose_line(line) for line in lines]
            det_lines = [line for line in det_lines if line]
            out_label = out_labels / label_path.name
            out_label.write_text(
                "\n".join(det_lines) + ("\n" if det_lines else ""),
                encoding="utf-8",
            )
            counts[f"labels_{split}"] += 1

            stem = label_path.stem
            image_src: Path | None = None
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidate = pose_images / f"{stem}{ext}"
                if candidate.is_file():
                    image_src = candidate
                    break
            if image_src is None:
                continue

            image_dst = out_images / image_src.name
            action = _link_or_copy(image_src, image_dst)
            if action != "skipped":
                counts[f"images_{split}"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create YOLO detect dataset from pose labels (box columns only)",
    )
    parser.add_argument(
        "--pose-dir",
        default="data/wildlife_bird",
        help="Source pose dataset root",
    )
    parser.add_argument(
        "--output",
        default="data/wildlife_bird_det",
        help="Output detect dataset root",
    )
    args = parser.parse_args()

    counts = make_detection_labels(Path(args.pose_dir), Path(args.output))
    print(f"Wrote detect dataset under {args.output}")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
