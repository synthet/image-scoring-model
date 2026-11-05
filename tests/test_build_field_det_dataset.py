"""Fast tests for the detector dataset builder: box conversion, date rule, non-bird and dropped frames."""
import json
import random

import pytest

from training.build_field_det_dataset import _cub_replay, check_held_out, field_rows, held_dates, shooting_date, yolo_line


def test_shooting_date_uses_the_folder_not_the_file_name():
    assert shooting_date("D:\\P\\Z8\\180-600mm\\2026\\2026-08-23\\DSC_1.NEF") == "2026-08-23"
    assert shooting_date("/mnt/d/P/Camera JPEG/2009-09-10/2011-01-01.jpg") == "2009-09-10"
    assert shooting_date("D:/P/Z8/DSC_1.NEF") is None


def test_yolo_line_normalises_and_clips_to_the_frame():
    assert yolo_line([100, 50, 300, 150], 400, 200) == "0 0.500000 0.500000 0.500000 0.500000"
    assert yolo_line([-100, 0, 200, 300], 400, 200) == "0 0.250000 0.500000 0.500000 1.000000"


def _sample(tmp_path, labels, subjects=None):
    sample = tmp_path / "s"
    sample.mkdir()
    ids = [row["image_id"] for row in labels]
    (sample / "manifest.jsonl").write_text("".join(json.dumps(
        {"image_id": i, "image_path": f"{i}.jpg", "source_path": f"D:/P/2020-01-01/{i}.NEF", "width": 100, "height": 100}) + "\n"
        for i in ids), encoding="utf-8")
    (sample / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
    if subjects:
        (sample / "subjects.jsonl").write_text("".join(json.dumps(row) + "\n" for row in subjects), encoding="utf-8")
    return sample


def test_non_bird_box_leaves_a_hard_negative(tmp_path):
    sample = _sample(tmp_path, [{"image_id": "a", "birds": [{"bbox": [0, 0, 9, 9]}]}],
                     [{"image_id": "a", "bird_index": 0, "subject": "other"}])
    assert field_rows(sample, None, set())[0]["boxes"] == []


def test_allow_drop_and_excluded_filter_frames(tmp_path):
    sample = _sample(tmp_path, [{"image_id": i, "birds": [], "excluded": i == "c"} for i in "abcd"])
    assert [r["id"] for r in field_rows(sample, {"a", "b", "c"}, {"b"})] == ["s_a"]


def test_build_stops_on_a_held_out_or_undated_frame(tmp_path):
    held = tmp_path / "held.jsonl"
    held.write_text(json.dumps({"source_path": "D:/P/Z8/2021-05-05/x.NEF"}) + "\n", encoding="utf-8")
    cohort = tmp_path / "cohort.csv"
    cohort.write_text("file_path\n/mnt/d/P/D90/2022-06-06/y.NEF\n", encoding="utf-8")
    dates = held_dates([str(held)], [str(cohort)])
    assert dates == {"2021-05-05", "2022-06-06"}
    check_held_out(["D:/P/Z6/2020-01-01/ok.NEF"], dates)
    with pytest.raises(SystemExit):
        check_held_out(["D:/P/Z6/2021-05-05/other_lens.NEF"], dates)
    with pytest.raises(SystemExit):
        check_held_out(["D:/P/Z6/undated.NEF"], dates)


def test_cub_replay_skips_cache_files_and_keeps_only_box_columns(tmp_path):
    cub = tmp_path / "cub"
    (cub / "images" / "val").mkdir(parents=True)
    (cub / "labels" / "val").mkdir(parents=True)
    (cub / "images" / "val" / "a.jpg").write_bytes(b"x")
    (cub / "images" / "val" / "a.npy").write_bytes(b"x")
    (cub / "labels" / "val" / "a.txt").write_text("0 0.5 0.5 0.2 0.2 0.1 0.1 2\n", encoding="utf-8")
    listed = _cub_replay(cub, "val", 10, tmp_path / "out", random.Random(0))
    assert [p.rsplit("/", 1)[-1] for p in listed] == ["a.jpg"]
    assert (tmp_path / "out" / "labels" / "cub_val" / "a.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
