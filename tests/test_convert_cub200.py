"""Tests for CUB-200 conversion."""

from training.convert_cub200 import map_cub_parts_to_keypoints


def test_map_cub_parts_produces_six_keypoints():
    part_ids = {
        1: "back",
        2: "beak",
        3: "belly",
        4: "breast",
        5: "crown",
        6: "forehead",
        7: "left eye",
        8: "left leg",
        9: "left wing",
        10: "nape",
        11: "right eye",
        12: "right leg",
        13: "right wing",
        14: "tail",
        15: "throat",
    }
    parts = {
        2: (120.0, 80.0, 1),
        6: (100.0, 60.0, 1),
        7: (90.0, 55.0, 1),
        11: (110.0, 55.0, 1),
        5: (98.0, 45.0, 1),
        9: (140.0, 90.0, 1),
        13: (60.0, 90.0, 1),
        10: (105.0, 75.0, 1),
    }
    bbox = (50.0, 30.0, 120.0, 100.0)
    kpts = map_cub_parts_to_keypoints(parts, part_ids, bbox)
    assert kpts is not None
    assert len(kpts) == 18
    assert kpts[2] == 2  # CUB visible -> COCO visible
    assert kpts[5] == 2
    assert kpts[8] == 2


def test_hidden_eye_bird_retains_bbox_target():
    part_ids = {2: "beak", 7: "left eye", 11: "right eye", 6: "forehead"}
    parts = {2: (120.0, 80.0, 1)}
    bbox = (50.0, 30.0, 120.0, 100.0)
    kpts = map_cub_parts_to_keypoints(parts, part_ids, bbox)
    assert kpts[:3] == [120.0, 80.0, 2]
    assert kpts[3:9] == [0.0] * 6


def test_head_only_annotation_does_not_create_visible_eyes():
    kpts = map_cub_parts_to_keypoints(
        {5: (98.0, 45.0, 1)}, {5: "crown"}, (50.0, 30.0, 120.0, 100.0)
    )
    assert kpts[3:9] == [0.0] * 6
    assert kpts[9:12] == [98.0, 45.0, 2]


def test_single_eye_preserves_coordinates_without_guessing_other_eye():
    kpts = map_cub_parts_to_keypoints(
        {7: (90.0, 55.0, 1), 11: (110.0, 55.0, 0)},
        {7: "left eye", 11: "right eye"}, (50.0, 30.0, 120.0, 100.0)
    )
    assert kpts[3:6] == [90.0, 55.0, 2]
    assert kpts[6:9] == [0.0, 0.0, 0]
