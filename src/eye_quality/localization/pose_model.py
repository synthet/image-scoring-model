"""YOLO pose localization wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eye_quality.localization.keypoint_schema import (
    BEAK_IDX,
    DEFAULT_THRESHOLDS,
    EYE_INDICES,
    HEAD_TOP_IDX,
    KEYPOINT_NAMES,
    LEFT_SHOULDER_IDX,
    RIGHT_SHOULDER_IDX,
    KeypointThresholds,
)
from eye_quality.schemas import EyeSide, EyeVisibility

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FINETUNED_WEIGHTS = _REPO_ROOT / "models" / "eye_pose_v0.pt"


def resolve_weights_path(weights: str | Path | None = None) -> str:
    """Prefer fine-tuned bird weights when present."""
    if weights is not None and str(weights).strip():
        return str(weights)
    if DEFAULT_FINETUNED_WEIGHTS.is_file():
        return str(DEFAULT_FINETUNED_WEIGHTS)
    logger.warning(
        "Fine-tuned weights not found at %s; falling back to yolo11n-pose.pt. "
        "Download eye_pose_v0.pt from https://huggingface.co/synthet/eye-pose-v0/tree/main "
        "or run training/convert_cub200.py and training/train_pose.py first.",
        DEFAULT_FINETUNED_WEIGHTS,
    )
    return "yolo11n-pose.pt"


DEFAULT_WEIGHTS = resolve_weights_path()


@dataclass
class PoseKeypoint:
    name: str
    x: float
    y: float
    confidence: float


@dataclass
class PoseDetection:
    bbox_xyxy: tuple[float, float, float, float]
    bbox_confidence: float
    keypoints: list[PoseKeypoint] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    @property
    def bbox_area_frac(self) -> float:
        if self.image_width <= 0 or self.image_height <= 0:
            return 0.0
        x1, y1, x2, y2 = self.bbox_xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return area / float(self.image_width * self.image_height)


@dataclass
class EyeLocalization:
    eye_side: EyeSide
    center_x: float
    center_y: float
    confidence: float
    visibility: EyeVisibility
    bbox_norm: list[float]
    head_width_px: float
    subject_area_frac: float


class PoseLocalizer:
    """Run Ultralytics YOLO pose and map wildlife keypoints to eye localizations."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        device: str | None = None,
        imgsz: int = 640,
        thresholds: KeypointThresholds | None = None,
    ) -> None:
        self.weights = resolve_weights_path(weights)
        self.device = device
        self.imgsz = imgsz
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        logger.info("Loading YOLO pose weights: %s", self.weights)
        self._model = YOLO(self.weights)
        return self._model

    def predict(self, image_path: str | Path) -> list[PoseDetection]:
        """Run pose inference; returns detections in pixel coordinates."""
        from PIL import Image

        image_path = Path(image_path)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size

        model = self._load_model()
        kwargs: dict[str, Any] = {"imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = model.predict(str(image_path), **kwargs)

        detections: list[PoseDetection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        kpts = keypoints.data.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            bbox_conf = float(confs[i])
            if bbox_conf < self.thresholds.min_bbox_conf:
                continue

            det = PoseDetection(
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_confidence=bbox_conf,
                image_width=width,
                image_height=height,
            )
            if i < len(kpts):
                det.keypoints = self._parse_keypoints(kpts[i])
            if det.bbox_area_frac < self.thresholds.min_bbox_area_frac:
                continue
            detections.append(det)

        detections.sort(key=lambda d: d.bbox_confidence, reverse=True)
        return detections

    def _parse_keypoints(self, row: np.ndarray) -> list[PoseKeypoint]:
        points: list[PoseKeypoint] = []
        n = min(len(KEYPOINT_NAMES), len(row))
        for idx in range(n):
            x, y, conf = float(row[idx, 0]), float(row[idx, 1]), float(row[idx, 2])
            points.append(
                PoseKeypoint(
                    name=KEYPOINT_NAMES[idx],
                    x=x,
                    y=y,
                    confidence=conf,
                )
            )
        return points

    def localize_eyes(
        self,
        image_path: str | Path,
        image_width: int,
        image_height: int,
    ) -> tuple[list[EyeLocalization], list[list[float]], list[str]]:
        """
        Return eye localizations, raw keypoint debug rows, and notes.
        """
        notes: list[str] = []
        detections = self.predict(image_path)
        if not detections:
            notes.append("no_pose_detection")
            return [], [], notes

        best = detections[0]
        kpt_map = {kp.name: kp for kp in best.keypoints}
        debug_kpts = [
            [kpt_map[name].x, kpt_map[name].y, kpt_map[name].confidence]
            if name in kpt_map
            else [0.0, 0.0, 0.0]
            for name in KEYPOINT_NAMES
        ]

        head_width = self._estimate_head_width(kpt_map, best.bbox_xyxy)
        turned_away = self._is_turned_away(kpt_map)

        localizations: list[EyeLocalization] = []
        for eye_side, idx in EYE_INDICES.items():
            name = KEYPOINT_NAMES[idx]
            kp = kpt_map.get(name)
            if kp is None:
                continue

            visibility, conf = self._eye_visibility(
                kp,
                head_width,
                turned_away,
                image_width,
                image_height,
            )
            if visibility in (EyeVisibility.HIDDEN, EyeVisibility.TOO_SMALL):
                if conf < self.thresholds.min_kpt_conf:
                    continue

            half = max(head_width * 0.12, 8.0)
            x1 = max(0.0, kp.x - half)
            y1 = max(0.0, kp.y - half)
            x2 = min(float(image_width), kp.x + half)
            y2 = min(float(image_height), kp.y + half)
            bbox_norm = [
                x1 / image_width,
                y1 / image_height,
                max(1.0, x2 - x1) / image_width,
                max(1.0, y2 - y1) / image_height,
            ]

            localizations.append(
                EyeLocalization(
                    eye_side=eye_side,  # type: ignore[arg-type]
                    center_x=kp.x,
                    center_y=kp.y,
                    confidence=conf,
                    visibility=visibility,
                    bbox_norm=bbox_norm,
                    head_width_px=head_width,
                    subject_area_frac=best.bbox_area_frac,
                )
            )

        if not localizations:
            notes.append("no_visible_eyes")
        return localizations, debug_kpts, notes

    def _estimate_head_width(
        self,
        kpt_map: dict[str, PoseKeypoint],
        bbox_xyxy: tuple[float, float, float, float],
    ) -> float:
        left = kpt_map.get("left_eye")
        right = kpt_map.get("right_eye")
        if left and right and left.confidence > 0 and right.confidence > 0:
            return max(abs(right.x - left.x), 16.0)

        ls = kpt_map.get("left_shoulder")
        rs = kpt_map.get("right_shoulder")
        if ls and rs and ls.confidence > 0 and rs.confidence > 0:
            return max(abs(rs.x - ls.x) * 0.35, 16.0)

        x1, _, x2, _ = bbox_xyxy
        return max((x2 - x1) * 0.4, 16.0)

    def _is_turned_away(self, kpt_map: dict[str, PoseKeypoint]) -> bool:
        beak = kpt_map.get(KEYPOINT_NAMES[BEAK_IDX])
        head_top = kpt_map.get(KEYPOINT_NAMES[HEAD_TOP_IDX])
        left_eye = kpt_map.get(KEYPOINT_NAMES[EYE_INDICES["left"]])
        right_eye = kpt_map.get(KEYPOINT_NAMES[EYE_INDICES["right"]])
        if not beak or beak.confidence < self.thresholds.min_kpt_conf:
            return False
        visible_eyes = [
            kp
            for kp in (left_eye, right_eye)
            if kp and kp.confidence >= self.thresholds.min_kpt_conf
        ]
        if len(visible_eyes) == 0:
            return True
        if head_top and head_top.confidence >= self.thresholds.min_kpt_conf:
            avg_eye_y = sum(kp.y for kp in visible_eyes) / len(visible_eyes)
            if beak.y < avg_eye_y - 5:
                return True
        return False

    def _eye_visibility(
        self,
        kp: PoseKeypoint,
        head_width: float,
        turned_away: bool,
        image_width: int,
        image_height: int,
    ) -> tuple[EyeVisibility, float]:
        conf = float(kp.confidence)
        if turned_away and conf < self.thresholds.low_confidence_threshold:
            return EyeVisibility.TURNED_AWAY, conf
        if conf < self.thresholds.min_kpt_conf:
            return EyeVisibility.HIDDEN, conf

        crop_radius = max(head_width * 0.12, self.thresholds.min_eye_crop_px / 2)
        crop_px = crop_radius * 2
        if crop_px < self.thresholds.min_eye_crop_px:
            return EyeVisibility.TOO_SMALL, conf * 0.5

        if kp.x < 0 or kp.y < 0 or kp.x > image_width or kp.y > image_height:
            return EyeVisibility.OCCLUDED, conf * 0.7

        if conf >= self.thresholds.low_confidence_threshold:
            return EyeVisibility.VISIBLE, conf
        return EyeVisibility.PARTIALLY_VISIBLE, conf
