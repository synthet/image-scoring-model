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
    assert kpts[2] == 1  # beak visible
    assert kpts[5] == 1  # left eye visible
    assert kpts[8] == 1  # right eye visible


def test_skip_when_no_eye_or_head():
    part_ids = {2: "beak", 7: "left eye", 11: "right eye", 6: "forehead"}
    parts = {2: (120.0, 80.0, 1)}
    bbox = (50.0, 30.0, 120.0, 100.0)
    assert map_cub_parts_to_keypoints(parts, part_ids, bbox) is None
