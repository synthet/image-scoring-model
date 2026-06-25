from eye_quality.schemas import EyeVisibility, FailureType, PerEyeMetrics
from eye_quality.scoring.aggregate import aggregate_eye_quality, fuse_focus_score, score_to_grade


def test_fuse_focus_score_bounded():
    score = fuse_focus_score(0.8, 0.7, 0.6, 1.0, 0.1)
    assert 0.0 <= score <= 1.0


def test_grade_mapping():
    assert score_to_grade(0.9) == 4
    assert score_to_grade(0.7) == 3
    assert score_to_grade(0.5) == 2
    assert score_to_grade(0.3) == 1
    assert score_to_grade(0.1) == 0


def test_low_confidence_no_failure_type():
    per_eye = [
        PerEyeMetrics(
            eye_side="left",
            visibility=EyeVisibility.PARTIALLY_VISIBLE,
            detection_confidence=0.3,
            bbox_norm=[0.1, 0.1, 0.05, 0.05],
            sharpness_score=0.1,
            clarity_score=0.1,
            edge_energy=0.1,
            focus_score=0.1,
            failure_type=FailureType.DEFOCUS,
            crop_width_px=32,
        )
    ]
    result = aggregate_eye_quality(per_eye)
    assert result.eye_quality.failure_type == FailureType.NONE
    assert result.eye_quality.confidence < 0.45
