"""Shared test helpers and synthetic fixtures."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from eye_quality.localization.pose_model import EyeLocalization
from eye_quality.schemas import EyeVisibility


def make_synthetic_bird_image(
    path: Path,
    *,
    width: int = 400,
    height: int = 300,
    sharp_eye: bool = True,
    blur_sigma: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Create a synthetic image with a high-contrast eye patch.

    Returns (left_eye_x, left_eye_y, right_eye_x, right_eye_y) in pixel coords.
    """
    rng = np.random.default_rng(42)
    base = rng.integers(80, 140, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(base, mode="RGB")

    lx, ly = int(width * 0.42), int(height * 0.35)
    rx, ry = int(width * 0.58), int(height * 0.35)

    arr = np.array(img)
    for cx, cy in ((lx, ly), (rx, ry)):
        r = 10
        y1, y2 = max(0, cy - r), min(height, cy + r)
        x1, x2 = max(0, cx - r), min(width, cx + r)
        patch = arr[y1:y2, x1:x2].astype(np.float32)
        if sharp_eye:
            patch += np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]][0][..., None] * 0.5
            patch[5:-5, 5:-5] += 80
        else:
            patch = patch.mean(axis=(0, 1), keepdims=True)
        if blur_sigma > 0:
            from PIL import ImageFilter

            pimg = Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8))
            pimg = pimg.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
            patch = np.array(pimg)
        arr[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(np.uint8)

    Image.fromarray(arr).save(path, quality=95)
    return float(lx), float(ly), float(rx), float(ry)


def make_blank_landscape(path: Path, width: int = 320, height: int = 200) -> None:
    grad = np.linspace(40, 180, width, dtype=np.uint8)
    arr = np.tile(grad, (height, 1))
    rgb = np.stack([arr, arr + 10, arr + 20], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def make_exif_rotated_image(path: Path) -> tuple[float, float]:
    """Save a portrait image with EXIF orientation 6 (90° CW). Returns eye coords in oriented space."""
    img = Image.new("RGB", (200, 100), color=(90, 90, 90))
    arr = np.array(img)
    ex, ey = 150, 40
    arr[ey - 8 : ey + 8, ex - 8 : ex + 8] = 220
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=img.getexif())
    # Orientation tag 6 = rotate 90 CW when displayed
    from PIL import ExifTags

    exif = img.getexif()
    for k, v in ExifTags.TAGS.items():
        if v == "Orientation":
            exif[k] = 6
            break
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    path.write_bytes(buf.getvalue())
    # After exif_transpose: 100x200, eye moves
    return 40.0, 50.0


def mock_localize(
    eyes: list[tuple[str, float, float, float, EyeVisibility]],
    head_width: float = 60.0,
):
    def _fn(_path: str, width: int, height: int):
        locs: list[EyeLocalization] = []
        debug_kpts = [[0.0, 0.0, 0.0] for _ in range(6)]
        for eye_side, x, y, conf, vis in eyes:
            half = max(head_width * 0.12, 8.0)
            x1 = max(0.0, x - half)
            y1 = max(0.0, y - half)
            x2 = min(float(width), x + half)
            y2 = min(float(height), y + half)
            bbox_norm = [
                x1 / width,
                y1 / height,
                max(1.0, x2 - x1) / width,
                max(1.0, y2 - y1) / height,
            ]
            locs.append(
                EyeLocalization(
                    eye_side=eye_side,  # type: ignore[arg-type]
                    center_x=x,
                    center_y=y,
                    confidence=conf,
                    visibility=vis,
                    bbox_norm=bbox_norm,
                    head_width_px=head_width,
                    subject_area_frac=0.05,
                )
            )
        return locs, debug_kpts, []

    return _fn
