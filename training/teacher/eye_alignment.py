"""Teacher-independent geometry and explicit partial bird-keypoint targets.

All source coordinates refer to an already EXIF-oriented rendition. Model adapters
must decode their own output representation before calling these helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from eye_quality.localization.keypoint_schema import FLIP_IDX, KEYPOINT_NAMES


@dataclass(frozen=True)
class CropTransform:
    source_size: tuple[int, int]
    crop_xyxy: tuple[float, float, float, float]
    input_size: tuple[int, int]
    scale_xy: tuple[float, float]
    padding_xy: tuple[float, float]

    @classmethod
    def from_crop(cls, source_size, crop_xyxy, input_size, *, letterbox=False):
        """Describe continuous-edge resize geometry; use actual adapter dimensions.

        This does not decode images or round crop edges. The adapter must supply
        the exact crop it rendered, not a proposed box before integer rounding.
        """
        w, h = source_size
        iw, ih = input_size
        x1, y1, x2, y2 = crop_xyxy
        if not np.isfinite([w, h, iw, ih, x1, y1, x2, y2]).all():
            raise ValueError("Crop geometry must be finite")
        if min(w, h, iw, ih) <= 0 or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("Crop must have positive dimensions and lie inside the oriented image")
        sx, sy = iw / (x2 - x1), ih / (y2 - y1)
        if letterbox:
            sx = sy = min(sx, sy)
        padding = ((iw - (x2 - x1) * sx) / 2, (ih - (y2 - y1) * sy) / 2)
        return cls(tuple(source_size), tuple(crop_xyxy), tuple(input_size), (sx, sy), padding)

    def to_input(self, points):
        return (self._points(points) - self.crop_xyxy[:2]) * self.scale_xy + self.padding_xy

    def to_source(self, points):
        return (self._points(points) - self.padding_xy) / self.scale_xy + self.crop_xyxy[:2]

    @staticmethod
    def _points(points):
        result = np.asarray(points, dtype=np.float64)
        if result.ndim < 1 or result.shape[-1] != 2 or not np.isfinite(result).all():
            raise ValueError("Points must be finite (..., 2) coordinates")
        return result


@dataclass(frozen=True)
class PartialKeypoints:
    """Normalized six-point targets; unknown and known-absent are distinct.

    The separate mask MUST reach both losses. Do not export just keypoints to a
    stock YOLO text file: that would silently turn unknowns into negatives.
    """

    keypoints: np.ndarray
    annotation_known: np.ndarray

    def horizontal_flip(self):
        points = self.keypoints[FLIP_IDX].copy()
        known = self.annotation_known[FLIP_IDX].copy()
        visible = known & (points[:, 2] > 0)
        points[visible, 0] = 1.0 - points[visible, 0]
        return PartialKeypoints(points, known)


def align_keypoints(
    points: Mapping[str, tuple[float, float, float]],
    mapping: Mapping[str, str | None],
    transform: CropTransform,
    *,
    min_confidence: float,
    confirmed_absent: set[str] | frozenset[str] = frozenset(),
) -> PartialKeypoints:
    """Map decoded input-pixel points using independently verified identities.

    No orientation-based identity guess is made here. Confidence must already be
    calibrated to [0, 1]. ``confirmed_absent`` requires explicit annotation, never
    a teacher's low score. Padding/out-of-crop points remain unknown.
    """
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    names = [name for name in mapping.values() if name is not None]
    if len(names) != len(set(names)):
        raise ValueError("Teacher to anatomical mapping must be one-to-one")
    if (set(names) | set(confirmed_absent)) - set(KEYPOINT_NAMES):
        raise ValueError("Unknown anatomical keypoint name")
    targets = np.zeros((len(KEYPOINT_NAMES), 3), dtype=np.float64)
    known = np.zeros(len(KEYPOINT_NAMES), dtype=bool)
    for name in confirmed_absent:
        known[KEYPOINT_NAMES.index(name)] = True
    for teacher_name, values in points.items():
        if len(values) != 3 or not np.isfinite(values).all() or not 0 <= values[2] <= 1:
            raise ValueError("Teacher points require finite x/y and calibrated confidence in [0, 1]")
        name = mapping.get(teacher_name)
        if name is None or values[2] < min_confidence:
            continue
        source = transform.to_source(values[:2])
        x1, y1, x2, y2 = transform.crop_xyxy
        if not (x1 <= source[0] <= x2 and y1 <= source[1] <= y2):
            continue
        if name in confirmed_absent:
            raise ValueError("Observed point conflicts with confirmed absence")
        index = KEYPOINT_NAMES.index(name)
        targets[index, :2] = source / transform.source_size
        targets[index, 2] = 2
        known[index] = True
    return PartialKeypoints(targets, known)
