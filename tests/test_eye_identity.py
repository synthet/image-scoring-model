import numpy as np

from training.teacher.eye_identity import suggest_teacher_mapping
from training.teacher.rtmpose_bird_eye import SimCCDecode


def _decoded(eye0, eye1, beak, orient_logits):
    return SimCCDecode(
        points={"eye0": eye0, "eye1": eye1, "beak": beak},
        orientation_logits=np.asarray(orient_logits, dtype=np.float64),
    )


def test_facing_camera_maps_image_x_to_anatomical_eyes():
    logits = [0.0, 0.0, 5.0, 0.0, 0.0]
    dec = _decoded((40, 128, 0.9), (200, 128, 0.9), (128, 200, 0.9), logits)
    m = suggest_teacher_mapping(dec, kpt_gate=0.3, orient_min_conf=0.4)
    assert m["beak"] == "beak"
    assert m["eye0"] == "right_eye"
    assert m["eye1"] == "left_eye"


def test_single_eye_leaves_identity_unknown():
    logits = [0.0, 0.0, 5.0, 0.0, 0.0]
    dec = _decoded((40, 128, 0.9), (200, 128, 0.1), (128, 200, 0.9), logits)
    m = suggest_teacher_mapping(dec, kpt_gate=0.3)
    assert m["eye0"] is None and m["eye1"] is None
