from pathlib import Path

from eye_quality.pipeline import score_image
from eye_quality.schemas import EyeQualityPipelineResult, EyeVisibility, FailureType
from conftest import make_blank_landscape, make_synthetic_bird_image, mock_localize


def test_no_eye_found(tmp_path: Path):
    img = tmp_path / "landscape.jpg"
    make_blank_landscape(img)
    result = score_image(img, localize_fn=mock_localize([]))
    assert result.eye_quality.eye_visible_count == 0
    assert result.eye_quality.quality_grade == 0
    assert result.eye_quality.failure_type == FailureType.NONE
    assert result.eye_quality.confidence == 0.0


def test_one_eye_found(tmp_path: Path):
    img = tmp_path / "bird.jpg"
    lx, ly, _, _ = make_synthetic_bird_image(img, sharp_eye=True)
    fn = mock_localize([("left", lx, ly, 0.88, EyeVisibility.VISIBLE)])
    result = score_image(img, localize_fn=fn)
    assert result.eye_quality.eye_visible_count == 1
    assert result.eye_quality.best_eye_side == "left"
    assert result.eye_quality.focus_score > 0.0


def test_two_eyes_best_selection(tmp_path: Path):
    img = tmp_path / "bird.jpg"
    lx, ly, rx, ry = make_synthetic_bird_image(img, sharp_eye=True)
    fn = mock_localize(
        [
            ("left", lx, ly, 0.55, EyeVisibility.VISIBLE),
            ("right", rx, ry, 0.92, EyeVisibility.VISIBLE),
        ]
    )
    result = score_image(img, localize_fn=fn)
    assert result.eye_quality.eye_visible_count == 2
    assert result.eye_quality.best_eye_side == "right"


def test_tiny_subject(tmp_path: Path):
    img = tmp_path / "bird.jpg"
    lx, ly, _, _ = make_synthetic_bird_image(img)
    fn = mock_localize([("left", lx, ly, 0.7, EyeVisibility.TOO_SMALL)])
    result = score_image(img, localize_fn=fn)
    assert result.eye_quality.eye_visible_count == 0


def test_low_confidence_no_auto_reject(tmp_path: Path):
    img = tmp_path / "bird.jpg"
    lx, ly, _, _ = make_synthetic_bird_image(img, sharp_eye=False, blur_sigma=3.0)
    fn = mock_localize([("left", lx, ly, 0.35, EyeVisibility.PARTIALLY_VISIBLE)])
    result = score_image(img, localize_fn=fn)
    assert result.eye_quality.failure_type == FailureType.NONE


def test_json_schema_stable(tmp_path: Path):
    img = tmp_path / "bird.jpg"
    lx, ly, _, _ = make_synthetic_bird_image(img)
    fn = mock_localize([("left", lx, ly, 0.85, EyeVisibility.VISIBLE)])
    result = score_image(img, localize_fn=fn)
    data = result.to_api_dict()
    restored = EyeQualityPipelineResult.model_validate(data)
    assert restored.eye_quality.model_name == "eye_quality_heuristic"


def test_heuristic_monotonicity(tmp_path: Path):
    """Sharper synthetic crop should score higher than a blurred crop."""
    import numpy as np
    from eye_quality.scoring.heuristics import compute_eye_metrics

    rng = np.random.default_rng(0)
    sharp = rng.integers(80, 180, size=(48, 48), dtype=np.uint8)
    sharp[20:28, 20:28] = np.linspace(50, 255, 8, dtype=np.uint8)
    soft = sharp.copy()
    # Box blur approximation
    soft[18:30, 18:30] = int(np.mean(sharp[18:30, 18:30]))

    sharp_m = compute_eye_metrics(sharp, 48)
    soft_m = compute_eye_metrics(soft, 48)
    assert sharp_m["sharpness_score"] >= soft_m["sharpness_score"]
