import numpy as np

from eye_quality.scoring.heuristics import compute_eye_metrics


def test_sharp_patch_scores_higher_than_flat():
    sharp = np.tile(np.linspace(0, 255, 48, dtype=np.uint8), (48, 1))
    sharp += np.random.default_rng(0).integers(0, 30, size=sharp.shape, dtype=np.uint8)
    flat = np.full((48, 48), 128, dtype=np.uint8)

    sharp_m = compute_eye_metrics(sharp, 48)
    flat_m = compute_eye_metrics(flat, 48)

    assert sharp_m["sharpness_score"] > flat_m["sharpness_score"]
    assert sharp_m["edge_energy"] > flat_m["edge_energy"]


def test_empty_crop_returns_hidden():
    empty = np.zeros((1, 1), dtype=np.uint8)
    m = compute_eye_metrics(empty, 0)
    assert m["sharpness_score"] == 0.0
    assert m["size_factor"] == 0.0
