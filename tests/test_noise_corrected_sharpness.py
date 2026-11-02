"""Noise-corrected eye sharpness (model#8).

Plain Laplacian variance rises with sensor noise, so a defocused high-ISO eye can score as sharp. The
noise-corrected statistic (ported from image-scoring-pipeline ``modules/focus_quality.py``) removes the
``20 * sigma^2`` white-noise term, so added noise must not raise it.
"""

import numpy as np

from eye_quality.scoring import heuristics as h


def _defocused_eye(size=96, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    eye = 128 + 80 * np.exp(-(((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2 * (size / 6) ** 2)))
    eye += rng.normal(0, 0.5, eye.shape)  # a little texture, far below sensor noise
    # Defocus: repeated box blur.
    for _ in range(6):
        eye = (eye + np.roll(eye, 1, 0) + np.roll(eye, -1, 0) + np.roll(eye, 1, 1) + np.roll(eye, -1, 1)) / 5
    return eye


def _with_noise(img, sigma, seed=1):
    return np.clip(img + np.random.default_rng(seed).normal(0, sigma, img.shape), 0, 255)


def test_plain_laplacian_score_rises_with_noise_on_a_defocused_eye():
    # Documents the defect this change addresses; the default score is unchanged.
    soft = _defocused_eye()
    assert h.compute_eye_metrics(_with_noise(soft, 15), 96)["sharpness_score"] > (
        h.compute_eye_metrics(soft, 96)["sharpness_score"] + 0.2
    )


def _sharp_eye():
    sharp = _defocused_eye()
    rng = np.random.default_rng(2)
    sharp[40:56, 40:56] += rng.choice([-60.0, 60.0], size=(16, 16))  # crisp iris/catchlight detail
    return sharp


def test_noise_corrected_sharpness_removes_almost_all_of_the_noise_rise():
    soft = _defocused_eye()
    clean_plain, clean_corrected = h.compute_eye_metrics(soft, 96)["laplacian_var"], h.noise_corrected_sharpness(soft)
    sharp = h.noise_corrected_sharpness(_sharp_eye())
    for sigma in (5, 10, 15):
        noisy = _with_noise(soft, sigma)
        plain_rise = h.compute_eye_metrics(noisy, 96)["laplacian_var"] - clean_plain
        corrected_rise = h.noise_corrected_sharpness(noisy) - clean_corrected
        assert corrected_rise < 0.05 * plain_rise, (sigma, plain_rise, corrected_rise)
        # A noisy defocused eye stays far below a genuinely sharp one.
        assert h.noise_corrected_sharpness(noisy) < 0.1 * sharp, (sigma, sharp)


def test_noise_corrected_sharpness_still_separates_sharp_from_soft():
    soft = h.noise_corrected_sharpness(_defocused_eye())
    assert h.noise_corrected_sharpness(_sharp_eye()) > 5 * max(soft, 1.0)


def test_metrics_report_the_new_statistic_without_changing_the_default():
    soft = _defocused_eye()
    metrics = h.compute_eye_metrics(soft, 96)
    assert "sharpness_noise_corrected" in metrics
    assert metrics["sharpness_score"] == h._normalize_laplacian(metrics["laplacian_var"])
