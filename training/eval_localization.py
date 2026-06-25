#!/usr/bin/env python3
"""Quick localization QA: run the pose model over a split and report coverage.

Cheap regression signal for the Stage-1 eye localizer. Reports detection rate,
visible-eye rate, no-eye rate, and mean keypoint/bbox confidence so weight
updates can be compared apples-to-apples. Not a substitute for OKS/mAP (use
Ultralytics `val` for that) -- this measures end-to-end localize_eyes() output,
which is what the scoring pipeline actually consumes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eye_quality.crop.eye_crop import load_oriented_image
from eye_quality.localization.pose_model import PoseLocalizer, resolve_weights_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Localization coverage QA")
    parser.add_argument(
        "--images",
        default="data/wildlife_bird/images/val",
        help="Directory of images to evaluate (recursive)",
    )
    parser.add_argument("--weights", default=None, help="Pose weights (default: resolve_weights_path)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="Max images (0 = all)")
    parser.add_argument("--output", default=None, help="Write JSON summary to this path")
    args = parser.parse_args()

    root = Path(args.images)
    images = iter_images(root)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found under {root}")

    weights = resolve_weights_path(args.weights)
    localizer = PoseLocalizer(weights=weights, device=args.device, imgsz=args.imgsz)

    total = len(images)
    with_detection = 0
    with_visible_eye = 0
    no_pose = 0
    no_visible_eyes = 0
    visible_eye_counts: list[int] = []
    eye_confs: list[float] = []
    visibility_hist: Counter[str] = Counter()

    for img_path in images:
        try:
            oriented = load_oriented_image(img_path)
        except Exception as exc:  # noqa: BLE001 - keep QA loop robust
            visibility_hist["load_error"] += 1
            print(f"[skip] {img_path.name}: {exc}")
            continue

        localizations, _kpts, notes = localizer.localize_eyes(
            img_path, oriented.width, oriented.height
        )
        if "no_pose_detection" in notes:
            no_pose += 1
        else:
            with_detection += 1
        if "no_visible_eyes" in notes:
            no_visible_eyes += 1

        visible = [
            loc
            for loc in localizations
            if loc.visibility.value in ("visible", "partially_visible", "occluded")
        ]
        if visible:
            with_visible_eye += 1
        visible_eye_counts.append(len(visible))
        for loc in localizations:
            visibility_hist[loc.visibility.value] += 1
            eye_confs.append(float(loc.confidence))

    def pct(n: int) -> float:
        return round(100.0 * n / total, 2)

    summary = {
        "weights": weights,
        "images_dir": str(root),
        "total_images": total,
        "detection_rate_pct": pct(with_detection),
        "visible_eye_rate_pct": pct(with_visible_eye),
        "no_pose_detection_pct": pct(no_pose),
        "no_visible_eyes_pct": pct(no_visible_eyes),
        "mean_visible_eyes_per_image": round(
            sum(visible_eye_counts) / total if total else 0.0, 3
        ),
        "mean_eye_confidence": round(
            sum(eye_confs) / len(eye_confs) if eye_confs else 0.0, 4
        ),
        "visibility_histogram": dict(visibility_hist),
    }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
