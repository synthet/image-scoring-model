import numpy as np
import pytest

from training.teacher.rtmpose_bird_eye import decode_simcc_outputs, preprocess_crop


def test_preprocess_crop_shape_and_dtype():
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[128, 64] = (255, 128, 64)
    x = preprocess_crop(rgb)
    assert x.shape == (1, 3, 256, 256)
    assert x.dtype == np.float32


def test_decode_simcc_picks_argmax_bins_and_visibility_calibration():
    sx = np.zeros((3, 8), dtype=np.float32)
    sy = np.zeros((3, 8), dtype=np.float32)
    sx[0, 4] = 1.0
    sy[0, 6] = 1.0
    vis = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    decoded = decode_simcc_outputs({"simcc_x": sx[None], "simcc_y": sy[None], "kpt_visibility": vis})
    assert decoded.points["eye0"][:2] == (2.0, 3.0)
    assert decoded.points["eye0"][2] == pytest.approx(0.757, rel=1e-3)


def test_decode_without_visibility_uses_simcc_mean():
    sx = np.zeros((3, 4), dtype=np.float32)
    sy = np.zeros((3, 4), dtype=np.float32)
    sx[1, 2] = sy[1, 2] = 0.8
    decoded = decode_simcc_outputs({"simcc_x": sx[None], "simcc_y": sy[None]})
    assert decoded.points["eye1"][2] == pytest.approx(0.8)
