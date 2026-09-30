"""Map teacher eye0/eye1 to anatomical names using orientation (supporting evidence only).

Conservative: ambiguous orientation or a single visible eye leaves eye identities unknown.
"""
from __future__ import annotations

import numpy as np

from training.teacher.rtmpose_bird_eye import SimCCDecode, TEACHER_POINT_NAMES

ORIENTATIONS = (
    "facing_left",
    "facing_right",
    "facing_camera",
    "facing_away",
    "facing_up_down",
)


def orientation_confidence(logits: np.ndarray | None) -> tuple[str | None, float]:
    if logits is None or logits.size < 5:
        return None, 0.0
    v = np.asarray(logits[:5], dtype=np.float64)
    exp = np.exp(v - v.max())
    idx = int(np.argmax(exp))
    return ORIENTATIONS[idx], float(exp.max() / exp.sum())


def suggest_teacher_mapping(
    decoded: SimCCDecode,
    *,
    kpt_gate: float,
    orient_min_conf: float = 0.4,
) -> dict[str, str | None]:
    """Return teacher_name -> anatomical name or None (unknown)."""
    mapping: dict[str, str | None] = {name: None for name in TEACHER_POINT_NAMES}
    points = decoded.points
    if points["beak"][2] >= kpt_gate:
        mapping["beak"] = "beak"

    e0, e1 = points["eye0"], points["eye1"]
    if e0[2] < kpt_gate or e1[2] < kpt_gate:
        return mapping

    orient, oconf = orientation_confidence(decoded.orientation_logits)
    if orient is None or oconf < orient_min_conf or orient in ("facing_away", "facing_up_down"):
        return mapping

    ordered = sorted(
        (("eye0", e0), ("eye1", e1)),
        key=lambda item: item[1][0],
    )
    low_key, high_key = ordered[0][0], ordered[1][0]

    if orient == "facing_camera":
        mapping[high_key] = "left_eye"
        mapping[low_key] = "right_eye"
    elif orient == "facing_right":
        mapping[high_key] = "left_eye"
        mapping[low_key] = "right_eye"
    elif orient == "facing_left":
        mapping[low_key] = "left_eye"
        mapping[high_key] = "right_eye"
    return mapping


def pad_crop_pixels(box_xyxy: np.ndarray, width: int, height: int, pad: float = 0.125) -> tuple[int, int, int, int]:
    """Padded head crop (default 12.5% margin): x, y, w, h in source pixels."""
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    ox = max(0, round(x1 - bw * pad))
    oy = max(0, round(y1 - bh * pad))
    cw = min(width - ox, round(bw * (1 + 2 * pad)))
    ch = min(height - oy, round(bh * (1 + 2 * pad)))
    return int(ox), int(oy), int(cw), int(ch)


def crop_resize_rgb(rgb: np.ndarray, x: int, y: int, w: int, h: int, size: int = 256) -> np.ndarray:
    from PIL import Image

    if w <= 0 or h <= 0:
        raise ValueError("Crop must have positive width and height")
    patch = rgb[y:y + h, x:x + w]
    return np.asarray(Image.fromarray(patch).resize((size, size), Image.BILINEAR))
