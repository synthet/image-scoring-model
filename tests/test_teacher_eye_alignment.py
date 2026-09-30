import numpy as np
import pytest

from training.teacher.eye_alignment import CropTransform, align_keypoints


@pytest.mark.parametrize("letterbox", [False, True])
def test_non_square_crop_round_trip(letterbox):
    transform = CropTransform.from_crop((1000, 600), (100, 50, 500, 250), (256, 256), letterbox=letterbox)
    points = np.array([[100, 50], [300, 150], [500, 250]], dtype=float)
    np.testing.assert_allclose(transform.to_source(transform.to_input(points)), points)
    np.testing.assert_allclose(transform.to_input(points)[1], [128, 128])


def test_letterbox_padding_is_removed_before_source_normalization():
    transform = CropTransform.from_crop((1000, 600), (100, 50, 500, 250), (256, 256), letterbox=True)
    target = align_keypoints({"eye0": (64, 96, 0.9)}, {"eye0": "right_eye"}, transform, min_confidence=0.5)
    np.testing.assert_allclose(target.keypoints[2], [0.2, 100 / 600, 2])
    assert target.annotation_known.tolist() == [False, False, True, False, False, False]


def test_unknown_identity_and_low_confidence_remain_unknown():
    transform = CropTransform.from_crop((256, 256), (0, 0, 256, 256), (256, 256))
    target = align_keypoints(
        {"eye0": (10, 20, 0.99), "eye1": (30, 20, 0.2)},
        {"eye0": None, "eye1": "right_eye"}, transform, min_confidence=0.5,
    )
    assert not target.annotation_known.any()
    assert not target.keypoints.any()


def test_absent_eye_is_distinct_from_unannotated_eye_and_flip_is_reversible():
    transform = CropTransform.from_crop((256, 256), (0, 0, 256, 256), (256, 256))
    target = align_keypoints(
        {"eye0": (64, 32, 0.9)}, {"eye0": "left_eye"}, transform,
        min_confidence=0.5, confirmed_absent={"right_eye"},
    )
    assert target.annotation_known[1:3].all()
    assert target.keypoints[2, 2] == 0
    flipped = target.horizontal_flip()
    np.testing.assert_allclose(flipped.keypoints[2], [0.75, 0.125, 2])
    np.testing.assert_allclose(flipped.horizontal_flip().keypoints, target.keypoints)
    np.testing.assert_array_equal(flipped.horizontal_flip().annotation_known, target.annotation_known)


def test_duplicate_anatomical_mapping_is_rejected():
    transform = CropTransform.from_crop((256, 256), (0, 0, 256, 256), (256, 256))
    with pytest.raises(ValueError, match="one-to-one"):
        align_keypoints({}, {"eye0": "left_eye", "eye1": "left_eye"}, transform, min_confidence=0.5)


def test_point_in_padding_is_unknown_not_clamped_to_bird():
    transform = CropTransform.from_crop((1000, 600), (100, 50, 500, 250), (256, 256), letterbox=True)
    target = align_keypoints({"eye0": (128, 10, 0.99)}, {"eye0": "left_eye"}, transform, min_confidence=0.5)
    assert not target.annotation_known.any()


@pytest.mark.parametrize("box", [(0, 0, 0, 10), (-1, 0, 10, 10), (0, 0, 257, 256)])
def test_invalid_crop_rejected(box):
    with pytest.raises(ValueError):
        CropTransform.from_crop((256, 256), box, (256, 256))
