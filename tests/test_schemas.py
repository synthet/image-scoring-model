from eye_quality.schemas import EyeQualityPipelineResult, EyeQualitySummary, FailureType


def test_api_dict_roundtrip():
    result = EyeQualityPipelineResult(
        eye_quality=EyeQualitySummary(
            best_eye_side="left",
            eye_visible_count=1,
            focus_score=0.82,
            sharpness_score=0.79,
            clarity_score=0.74,
            quality_grade=3,
            failure_type=FailureType.NONE,
            confidence=0.91,
        ),
        detections=[],
    )
    data = result.to_api_dict()
    assert data["eye_quality"]["focus_score"] == 0.82
    assert data["eye_quality"]["failure_type"] == "none"
    restored = EyeQualityPipelineResult.model_validate(data)
    assert restored.eye_quality.quality_grade == 3
