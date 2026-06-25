"""Native-resolution eye crop extraction with EXIF-safe orientation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class OrientedImage:
    array_rgb: np.ndarray
    width: int
    height: int
    source_path: str


@dataclass
class EyeCropResult:
    gray: np.ndarray
    crop_path: str | None
    crop_width_px: int
    crop_height_px: int
    center_x: float
    center_y: float


def load_oriented_image(image_path: str | Path) -> OrientedImage:
    """Load image applying EXIF orientation for analysis (does not modify source)."""
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        oriented = ImageOps.exif_transpose(img)
        rgb = oriented.convert("RGB")
        arr = np.array(rgb)
    h, w = arr.shape[:2]
    return OrientedImage(array_rgb=arr, width=w, height=h, source_path=str(image_path))


def crop_eye_region(
    oriented: OrientedImage,
    center_x: float,
    center_y: float,
    head_width_px: float,
    *,
    debug_dir: str | Path | None = None,
    eye_side: str = "left",
    source_tag: str | None = None,
) -> EyeCropResult:
    """
    Extract a square eye crop at native oriented resolution.

    Crop radius scales with head width and enforces a minimum pixel size.
    """
    min_side = 24
    radius = max(head_width_px * 0.12, min_side / 2)
    radius = max(radius, min_side / 2)

    cx = int(round(center_x))
    cy = int(round(center_y))
    x1 = max(0, cx - int(round(radius)))
    y1 = max(0, cy - int(round(radius)))
    x2 = min(oriented.width, cx + int(round(radius)))
    y2 = min(oriented.height, cy + int(round(radius)))

    if x2 <= x1 or y2 <= y1:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return EyeCropResult(
            gray=empty,
            crop_path=None,
            crop_width_px=0,
            crop_height_px=0,
            center_x=center_x,
            center_y=center_y,
        )

    crop_rgb = oriented.array_rgb[y1:y2, x1:x2]
    gray = _to_gray(crop_rgb)

    crop_path: str | None = None
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        tag = source_tag or hashlib.md5(oriented.source_path.encode()).hexdigest()[:8]
        crop_path = str(debug_dir / f"{tag}_{eye_side}_eye.jpg")
        Image.fromarray(crop_rgb).save(crop_path, quality=92)

    return EyeCropResult(
        gray=gray,
        crop_path=crop_path,
        crop_width_px=int(gray.shape[1]),
        crop_height_px=int(gray.shape[0]),
        center_x=center_x,
        center_y=center_y,
    )


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # ITU-R BT.601 luma
    return (
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ).astype(np.uint8)
