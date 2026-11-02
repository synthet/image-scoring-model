"""Fallback filtering must preserve constant intensity and brightness shifts."""
import numpy as np
import pytest

from eye_quality.scoring import heuristics


@pytest.mark.parametrize("level", [32, 64, 128, 255])
def test_no_noise_is_invented_on_a_flat_crop_without_opencv(monkeypatch, level):
    monkeypatch.setattr(heuristics, "cv2", None)
    metrics = heuristics.compute_eye_metrics(np.full((48, 48), level, dtype=np.uint8), 48)
    assert metrics["noise_penalty"] == pytest.approx(0.0)


def test_fallback_noise_is_invariant_to_an_unclipped_brightness_shift(monkeypatch):
    monkeypatch.setattr(heuristics, "cv2", None)
    patch = np.tile(np.arange(48, dtype=np.uint8), (48, 1))
    base = heuristics.compute_eye_metrics(patch, 48)["noise_penalty"]
    shifted = heuristics.compute_eye_metrics(patch + 128, 48)["noise_penalty"]
    assert shifted == pytest.approx(base)
