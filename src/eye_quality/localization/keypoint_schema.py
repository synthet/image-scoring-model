"""Wildlife bird head keypoint schema for YOLO pose."""

from __future__ import annotations

from dataclasses import dataclass

KEYPOINT_NAMES: tuple[str, ...] = (
    "beak",
    "left_eye",
    "right_eye",
    "head_top",
    "left_shoulder",
    "right_shoulder",
)

KPT_SHAPE = (len(KEYPOINT_NAMES), 3)

# Swap left/right pairs on horizontal flip augmentation.
FLIP_IDX: list[int] = [0, 2, 1, 3, 5, 4]

LEFT_EYE_IDX = KEYPOINT_NAMES.index("left_eye")
RIGHT_EYE_IDX = KEYPOINT_NAMES.index("right_eye")
BEAK_IDX = KEYPOINT_NAMES.index("beak")
HEAD_TOP_IDX = KEYPOINT_NAMES.index("head_top")
LEFT_SHOULDER_IDX = KEYPOINT_NAMES.index("left_shoulder")
RIGHT_SHOULDER_IDX = KEYPOINT_NAMES.index("right_shoulder")

EYE_INDICES: dict[str, int] = {
    "left": LEFT_EYE_IDX,
    "right": RIGHT_EYE_IDX,
}


@dataclass(frozen=True)
class KeypointThresholds:
    """Visibility and confidence gates for pose keypoints."""

    min_kpt_conf: float = 0.25
    min_bbox_conf: float = 0.25
    min_bbox_area_frac: float = 0.002
    min_eye_crop_px: int = 12
    low_confidence_threshold: float = 0.45


DEFAULT_THRESHOLDS = KeypointThresholds()


def keypoint_index(name: str) -> int:
    try:
        return KEYPOINT_NAMES.index(name)
    except ValueError as exc:
        raise KeyError(f"Unknown keypoint: {name}") from exc
