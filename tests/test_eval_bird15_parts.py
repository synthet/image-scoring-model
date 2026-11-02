from training.eval_bird15_parts import part_errors, summarise


def test_errors_are_normalised_by_box_diagonal_and_low_confidence_counts_as_missed():
    box = [0.5, 0.5, 0.3, 0.4]                    # 30x40 px box in a 100x100 image -> diagonal 50 px
    kp = [(0.5, 0.5, 2)] + [(0.0, 0.0, 0)] * 13 + [(0.2, 0.2, 2)]
    pred = [[55.0, 50.0]] + [[0.0, 0.0]] * 14
    conf = [0.9] + [0.9] * 13 + [0.1]
    errs = part_errors(pred, conf, box, kp, 100, 100, 0.5)
    assert set(errs) == {0, 14}                    # only visible parts are scored
    assert abs(errs[0] - 0.1) < 1e-9               # 5 px / 50 px
    assert errs[14] is None                        # low confidence = miss


def test_summary_counts_misses_against_pck():
    rows = summarise({0: [0.04, 0.12, None, 0.3]})
    assert rows["back"]["visible"] == 4
    assert rows["back"]["pck@0.05"] == 0.25
    assert rows["back"]["pck@0.2"] == 0.5
    assert rows["beak"]["visible"] == 0
