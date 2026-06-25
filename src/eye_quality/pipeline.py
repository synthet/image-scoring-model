"""End-to-end eye quality pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from eye_quality.crop.eye_crop import crop_eye_region, load_oriented_image
from eye_quality.localization.pose_model import EyeLocalization, PoseLocalizer, resolve_weights_path
from eye_quality.schemas import (
    DebugInfo,
    EyeQualityPipelineResult,
    EyeVisibility,
    FailureType,
    PerEyeMetrics,
)
from eye_quality.scoring.aggregate import aggregate_eye_quality, fuse_focus_score
from eye_quality.scoring.heuristics import compute_eye_metrics

LocalizeFn = Callable[[str, int, int], tuple[list[EyeLocalization], list[list[float]], list[str]]]


@dataclass
class PipelineConfig:
    weights: str = ""
    device: str | None = None
    imgsz: int = 640
    debug_dir: str | Path | None = None
    model_name: str = "eye_quality_heuristic"
    model_version: str = "0.1.0"

    def resolved_weights(self) -> str:
        return resolve_weights_path(self.weights or None)


def score_image(
    image_path: str | Path,
    *,
    config: PipelineConfig | None = None,
    localizer: PoseLocalizer | None = None,
    localize_fn: LocalizeFn | None = None,
) -> EyeQualityPipelineResult:
    """
    Localize eyes, crop at native resolution, score heuristics, aggregate.

    ``localize_fn`` overrides YOLO inference (used in unit tests).
    """
    config = config or PipelineConfig()
    image_path = Path(image_path)
    oriented = load_oriented_image(image_path)

    debug = DebugInfo()
    if localize_fn is not None:
        localizations, debug_kpts, notes = localize_fn(
            str(image_path), oriented.width, oriented.height
        )
    else:
        loc = localizer or PoseLocalizer(
            weights=config.resolved_weights(),
            device=config.device,
            imgsz=config.imgsz,
        )
        localizations, debug_kpts, notes = loc.localize_eyes(
            image_path, oriented.width, oriented.height
        )

    debug.pose_keypoints = debug_kpts
    debug.notes.extend(notes)

    per_eye: list[PerEyeMetrics] = []
    for loc in localizations:
        crop = crop_eye_region(
            oriented,
            loc.center_x,
            loc.center_y,
            loc.head_width_px,
            debug_dir=config.debug_dir,
            eye_side=loc.eye_side,
            source_tag=image_path.stem,
        )
        metrics = compute_eye_metrics(crop.gray, crop.crop_width_px)
        failure = metrics["failure_type"]
        if not isinstance(failure, FailureType):
            failure = FailureType.NONE

        focus = fuse_focus_score(
            float(metrics["sharpness_score"]),
            float(metrics["clarity_score"]),
            float(metrics["edge_energy"]),
            float(metrics["size_factor"]),
            float(metrics["noise_penalty"]),
        )

        per_eye.append(
            PerEyeMetrics(
                eye_side=loc.eye_side,
                visibility=loc.visibility,
                detection_confidence=loc.confidence,
                bbox_norm=loc.bbox_norm,
                sharpness_score=float(metrics["sharpness_score"]),
                clarity_score=float(metrics["clarity_score"]),
                edge_energy=float(metrics["edge_energy"]),
                focus_score=focus,
                noise_penalty=float(metrics["noise_penalty"]),
                size_factor=float(metrics["size_factor"]),
                failure_type=failure,
                crop_path=crop.crop_path,
                crop_width_px=crop.crop_width_px,
            )
        )

    return aggregate_eye_quality(
        per_eye,
        model_name=config.model_name,
        model_version=config.model_version,
        debug=debug,
    )


def empty_result(
    *,
    model_name: str = "eye_quality_heuristic",
    model_version: str = "0.1.0",
    notes: list[str] | None = None,
) -> EyeQualityPipelineResult:
    """No-op result for missing files or skipped images."""
    debug = DebugInfo(notes=notes or ["skipped"])
    return aggregate_eye_quality([], model_name=model_name, model_version=model_version, debug=debug)
