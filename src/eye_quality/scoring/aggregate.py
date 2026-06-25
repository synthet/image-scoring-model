"""Aggregate per-eye metrics into image-level eye quality summary."""

from __future__ import annotations

from eye_quality.schemas import (
    DebugInfo,
    EyeDetection,
    EyeQualityPipelineResult,
    EyeQualitySummary,
    EyeVisibility,
    FailureType,
    PerEyeMetrics,
)

CONFIDENCE_GATE = 0.45
MIN_USABLE_CROP_PX = 24


def score_to_grade(focus_score: float) -> int:
    if focus_score >= 0.85:
        return 4
    if focus_score >= 0.65:
        return 3
    if focus_score >= 0.45:
        return 2
    if focus_score >= 0.20:
        return 1
    return 0


def fuse_focus_score(
    sharpness: float,
    clarity: float,
    edge_energy: float,
    size_factor: float,
    noise_penalty: float,
) -> float:
    raw = 0.45 * sharpness + 0.35 * clarity + 0.20 * edge_energy
    raw *= size_factor
    raw *= max(0.0, 1.0 - noise_penalty)
    return float(max(0.0, min(1.0, raw)))


def infer_blur_failure(focus_score: float, sharpness: float) -> FailureType:
    if sharpness < 0.15 and focus_score < 0.25:
        return FailureType.DEFOCUS
    if focus_score < 0.20:
        return FailureType.MOTION_BLUR
    return FailureType.NONE


def aggregate_eye_quality(
    per_eye: list[PerEyeMetrics],
    *,
    model_name: str = "eye_quality_heuristic",
    model_version: str = "0.1.0",
    debug: DebugInfo | None = None,
) -> EyeQualityPipelineResult:
    """Build pipeline result from per-eye metric rows."""
    debug = debug or DebugInfo()
    detections: list[EyeDetection] = []

    visible = [
        m
        for m in per_eye
        if m.visibility
        not in (EyeVisibility.HIDDEN, EyeVisibility.TOO_SMALL, EyeVisibility.TURNED_AWAY)
    ]

    for m in per_eye:
        detections.append(
            EyeDetection(
                eye_side=m.eye_side,
                bbox_norm=m.bbox_norm,
                visibility=m.visibility,
                detection_confidence=m.detection_confidence,
                crop_path=m.crop_path,
            )
        )
        if m.crop_path:
            debug.crop_paths.append(m.crop_path)

    if not visible:
        summary = EyeQualitySummary(
            best_eye_side=None,
            eye_visible_count=0,
            focus_score=0.0,
            sharpness_score=0.0,
            clarity_score=0.0,
            quality_grade=0,
            failure_type=FailureType.NONE,
            confidence=0.0,
            model_name=model_name,
            model_version=model_version,
        )
        return EyeQualityPipelineResult(
            eye_quality=summary,
            detections=detections,
            debug=debug,
        )

    scored = sorted(
        visible,
        key=lambda m: (m.focus_score, m.sharpness_score),
        reverse=True,
    )
    best = scored[0]

    confidence = float(best.detection_confidence)
    if best.crop_width_px < MIN_USABLE_CROP_PX:
        confidence *= 0.5

    failure = FailureType.NONE
    if confidence >= CONFIDENCE_GATE:
        if best.visibility == EyeVisibility.TURNED_AWAY:
            failure = FailureType.TURNED_AWAY
        elif best.visibility == EyeVisibility.TOO_SMALL or best.size_factor < 0.4:
            failure = FailureType.SUBJECT_TOO_SMALL
        elif best.failure_type != FailureType.NONE:
            failure = best.failure_type
        else:
            blur_fail = infer_blur_failure(best.focus_score, best.sharpness_score)
            if blur_fail != FailureType.NONE:
                failure = blur_fail
            elif best.noise_penalty > 0.6:
                failure = FailureType.NOISE

    grade = score_to_grade(best.focus_score)
    if confidence < CONFIDENCE_GATE:
        failure = FailureType.NONE
        grade = min(grade, 2)

    summary = EyeQualitySummary(
        best_eye_side=best.eye_side,
        eye_visible_count=len(visible),
        focus_score=best.focus_score,
        sharpness_score=best.sharpness_score,
        clarity_score=best.clarity_score,
        quality_grade=grade,
        failure_type=failure,
        confidence=confidence,
        model_name=model_name,
        model_version=model_version,
    )
    return EyeQualityPipelineResult(
        eye_quality=summary,
        detections=detections,
        debug=debug,
    )
