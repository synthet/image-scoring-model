"""GPU detection tests (require models/bird_detect_v0.pt + CUDA)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "models" / "bird_detect_v0.pt"
VAL_IMAGES = REPO_ROOT / "data" / "wildlife_bird_det" / "images" / "val"
VAL_LABELS = REPO_ROOT / "data" / "wildlife_bird_det" / "labels" / "val"


def _yolo_norm_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    bw, bh = w * img_w, h * img_h
    x1 = (cx * img_w) - bw / 2
    y1 = (cy * img_h) - bh / 2
    return x1, y1, x1 + bw, y1 + bh


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@pytest.mark.gpu
def test_bird_detect_iou_on_val_image():
    if not WEIGHTS.is_file():
        pytest.skip(f"weights not found: {WEIGHTS}")

    val_images = sorted(VAL_IMAGES.glob("*.jpg"))
    if not val_images:
        pytest.skip("no val images in data/wildlife_bird_det/images/val")

    image_path = val_images[0]
    label_path = VAL_LABELS / f"{image_path.stem}.txt"
    if not label_path.is_file():
        pytest.skip(f"no label for {image_path.name}")

    from PIL import Image

    from eye_quality.localization.bird_detector import BirdDetector

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    parts = label_path.read_text(encoding="utf-8").strip().split()
    assert len(parts) >= 5
    cx, cy, w, h = (float(v) for v in parts[1:5])
    gt = _yolo_norm_to_xyxy(cx, cy, w, h, img_w, img_h)

    detector = BirdDetector(weights=WEIGHTS, device="0")
    boxes = detector.predict(image_path, conf=0.25)
    assert boxes, "expected at least one bird detection"
    best = boxes[0]
    assert best.confidence > 0.5
    assert _iou(best.bbox_xyxy, gt) > 0.5
