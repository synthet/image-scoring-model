"""Unit tests for training/compare_eye_benchmark.py (no weights needed)."""
from __future__ import annotations

import json

from training.compare_eye_benchmark import compare, paired_difference
from training.eval_eye_benchmark import frame_records

BOX = [0, 0, 300, 400]


def bird_frame(image_id):
    eyes = [{"state": "visible", "x": 100, "y": 100}, {"state": "hidden"}]
    return {"image_id": image_id, "kind": "bird", "size_stratum": "small", "body": "Z8",
            "birds": [{"id": "b", "bbox": BOX, "eyes": eyes}]}


def found(image_id):
    eyes = [{"x": 100, "y": 100, "confidence": 0.9, "side": None}]
    return {"image_id": image_id, "detections": [{"bbox": BOX, "confidence": 0.9, "eyes": eyes}]}


def recs(frames, hits):
    return [frame_records(f, found(f["image_id"]) if hit else {}, 0.8, 0.25) for f, hit in zip(frames, hits)]


def test_identical_runs_have_a_zero_width_interval():
    """Both runs resample the same frames, so a varied set still gives exactly zero."""
    frames = [bird_frame(f"f{i}") for i in range(10)]
    base = recs(frames, [i % 2 == 0 for i in range(10)])
    assert paired_difference(base, base, "bird_recall", n=200) == (0.0, 0.0, 0.0)


def test_candidate_losing_every_bird_gives_a_negative_interval():
    frames = [bird_frame(f"f{i}") for i in range(10)]
    diff, low, high = paired_difference(recs(frames, [True] * 10), recs(frames, [False] * 10), "bird_recall", n=200)
    assert (diff, low, high) == (-1.0, -1.0, -1.0)


def test_interval_brackets_a_partial_loss():
    frames = [bird_frame(f"f{i}") for i in range(20)]
    base = recs(frames, [True] * 20)
    cand = recs(frames, [i >= 5 for i in range(20)])
    diff, low, high = paired_difference(base, cand, "bird_recall", n=500)
    assert diff == -0.25 and low < diff < high <= 0.0


def test_undefined_samples_are_skipped():
    """A resample holding only the hidden-eye frame has no visible eyes, so pck is None there."""
    hidden = bird_frame("f1")
    hidden["birds"][0]["eyes"] = [{"state": "hidden"}, {"state": "hidden"}]
    frames = [bird_frame("f0"), hidden]
    base = recs(frames, [True, True])
    assert paired_difference(base, base, "pck@0.05", n=50) == (0.0, 0.0, 0.0)


def test_compare_reads_saved_predictions(tmp_path):
    frames = [bird_frame(f"f{i}") for i in range(4)]
    for name, hits in (("base", [True] * 4), ("cand", [True, True, True, False])):
        (tmp_path / name).mkdir()
        rows = [found(f["image_id"]) for f, hit in zip(frames, hits) if hit]
        (tmp_path / name / "P_crop.predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                                                  encoding="utf-8")
    rows = compare(frames, tmp_path / "base", tmp_path / "cand", "P_crop", floor=0.8, bird_conf=0.25,
                   keys=("bird_recall",), n=50)
    assert rows[0]["metric"] == "bird_recall"
    assert (rows[0]["base"], rows[0]["candidate"], rows[0]["difference"]) == (1.0, 0.75, -0.25)
