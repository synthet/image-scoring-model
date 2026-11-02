from training.convert_cub200_bird15 import BIRD15_FLIP_IDX, BIRD15_NAMES, yolo_line


def test_flip_index_swaps_left_and_right_parts_only():
    for i, name in enumerate(BIRD15_NAMES):
        mirrored = BIRD15_NAMES[BIRD15_FLIP_IDX[i]]
        expected = name.replace("left", "right") if "left" in name else name.replace("right", "left")
        assert mirrored == expected
    assert sorted(BIRD15_FLIP_IDX) == list(range(15))


def test_yolo_line_normalises_box_and_marks_missing_parts():
    line = yolo_line((10.0, 20.0, 100.0, 50.0), {9: (60.0, 40.0, 1)}, 200, 100).split()
    assert line[0] == "0"
    assert [float(v) for v in line[1:5]] == [0.3, 0.45, 0.5, 0.5]
    kp = line[5:]
    assert len(kp) == 45
    assert kp[0:3] == ["0.000000", "0.000000", "0"]          # back missing
    assert kp[24:27] == ["0.300000", "0.400000", "2"]        # left_wing (part 9) observed
