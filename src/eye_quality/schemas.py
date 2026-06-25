"""Pydantic models for eye quality pipeline output (backend contract)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EyeVisibility(str, Enum):
    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    HIDDEN = "hidden"
    TURNED_AWAY = "turned_away"
    TOO_SMALL = "too_small"
    OCCLUDED = "occluded"


class FailureType(str, Enum):
    NONE = "none"
    DEFOCUS = "defocus"
    MOTION_BLUR = "motion_blur"
    SUBJECT_MOTION = "subject_motion"
    NOISE = "noise"
    LOW_CONTRAST = "low_contrast"
    OVEREXPOSED = "overexposed"
    HIDDEN_EYE = "hidden_eye"
    TURNED_AWAY = "turned_away"
    SUBJECT_TOO_SMALL = "subject_too_small"


EyeSide = Literal["left", "right"]


class EyeDetection(BaseModel):
    eye_side: EyeSide
    bbox_norm: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Normalized [x, y, width, height] in oriented image space",
    )
    visibility: EyeVisibility
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    crop_path: str | None = None


class EyeQualitySummary(BaseModel):
    best_eye_side: EyeSide | None = None
    eye_visible_count: int = Field(0, ge=0, le=2)
    focus_score: float = Field(0.0, ge=0.0, le=1.0)
    sharpness_score: float = Field(0.0, ge=0.0, le=1.0)
    clarity_score: float = Field(0.0, ge=0.0, le=1.0)
    quality_grade: int = Field(0, ge=0, le=4)
    failure_type: FailureType = FailureType.NONE
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    model_name: str = "eye_quality_heuristic"
    model_version: str = "0.1.0"


class DebugInfo(BaseModel):
    crop_paths: list[str] = Field(default_factory=list)
    pose_keypoints: list[list[float]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EyeQualityPipelineResult(BaseModel):
    eye_quality: EyeQualitySummary
    detections: list[EyeDetection] = Field(default_factory=list)
    debug: DebugInfo = Field(default_factory=DebugInfo)

    def to_api_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PerEyeMetrics(BaseModel):
    eye_side: EyeSide
    visibility: EyeVisibility
    detection_confidence: float
    bbox_norm: list[float]
    sharpness_score: float = 0.0
    clarity_score: float = 0.0
    edge_energy: float = 0.0
    focus_score: float = 0.0
    noise_penalty: float = 0.0
    size_factor: float = 1.0
    failure_type: FailureType = FailureType.NONE
    crop_path: str | None = None
    crop_width_px: int = 0
