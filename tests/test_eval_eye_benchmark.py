"""Unit tests for training/eval_eye_benchmark.py (no weights needed)."""
from __future__ import annotations

import json
import random

import pytest

from training.eval_eye_benchmark import (
    aggregate,
    auroc,
    bootstrap,
    choose_threshold,
    eye_pose_v0_val_species,
    field_frames,
    frame_records,
    match_birds,
    match_eyes,
    parse_model_spec,
)


def bird(bbox, eyes):
    return {"id": "b", "bbox": bbox, "eyes": eyes}


def frame(birds, kind="bird", image_id="f1"):
    return {"image_id": image_id, "kind": kind, "size_stratum": "small", "body": "Z8", "birds": birds}


def det(bbox, eyes, conf=0.9):
    return {"bbox": bbox, "confidence": conf, "eyes": [{"x": x, "y": y, "confidence": c, "side": s} for x, y, c, s in eyes]}


BOX = [0, 0, 300, 400]  # diagonal 500


def test_match_eyes_is_order_independent():
    gt = [(10, 10), (100, 10)]
    pred = [(101, 12), (9, 10)]
    pairs = sorted(match_eyes(gt, pred))
    assert [(g, p) for g, p, _ in pairs] == [(0, 1), (1, 0)]
    assert match_eyes(gt, []) == []
    # one prediction pairs with the nearer truth
    assert [(g, p) for g, p, _ in match_eyes(gt, [(98, 11)])] == [(1, 0)]


def test_match_birds_one_to_one_by_iou():
    gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
    pred = [[21, 21, 30, 30], [0, 0, 9, 10], [0, 0, 10, 9]]
    assert match_birds(gt, pred) == {0: 1, 1: 0}
    assert match_birds(gt, [[100, 100, 110, 110]]) == {}


def test_frame_records_errors_misses_and_hidden():
    f = frame([bird(BOX, [{"state": "visible", "x": 100, "y": 100}, {"state": "hidden"}])])
    # model puts a confident eye 25 px from the truth (0.05 of the diagonal) and a second one on the hidden side
    p = {"detections": [det(BOX, [(100, 125, 0.9, None), (250, 100, 0.8, None)])]}
    r = frame_records(f, p, eye_conf=0.5, bird_conf=0.25)
    assert r["birds_found"] == 1
    assert r["eyes"][0]["error"] == pytest.approx(0.05)
    assert r["false_eyes_on_hidden"] == 1 and r["hidden_slots"] == 1
    # raise the gate above the extra eye: no false eye any more
    r2 = frame_records(f, p, eye_conf=0.85, bird_conf=0.25)
    assert r2["false_eyes_on_hidden"] == 0
    # missed bird: visible eye counts as not found
    r3 = frame_records(f, {"detections": []}, eye_conf=0.5, bird_conf=0.25)
    assert r3["birds_found"] == 0 and r3["eyes"] == [{"error": None, "confidence": None, "closed": False}]


def test_unknown_and_unresolvable_slots_are_not_scored():
    f = frame([bird(BOX, [{"state": "unknown"}, {"state": "unresolvable"}])])
    r = frame_records(f, {"detections": [det(BOX, [(1, 1, 0.9, None)])]}, 0.5, 0.25)
    assert r["eyes"] == [] and r["hidden_slots"] == 0 and r["false_eyes_on_hidden"] == 0


def test_negative_frame_counts_false_birds_and_eyes():
    f = frame([], kind="negative")
    p = {"detections": [det([0, 0, 10, 10], [(5, 5, 0.9, None), (6, 6, 0.1, None)]), det([20, 20, 30, 30], [], 0.1)]}
    r = frame_records(f, p, eye_conf=0.5, bird_conf=0.25)
    assert r["false_birds"] == 1 and r["negative_false_eyes"] == 1


def test_identity_scored_only_when_truth_has_sides():
    f = frame([bird(BOX, [{"state": "visible", "x": 100, "y": 100, "side": "left"},
                          {"state": "visible", "x": 200, "y": 100, "side": "right"}])])
    p = {"detections": [det(BOX, [(100, 101, 0.9, "left"), (199, 100, 0.9, None)])]}
    agg = aggregate([frame_records(f, p, 0.5, 0.25)])
    assert agg["side_correct_rate"] == 0.5 and agg["side_unassigned_rate"] == 0.5


def test_aggregate_pck_and_hit_rate():
    f = frame([bird(BOX, [{"state": "visible", "x": 100, "y": 100}, {"state": "visible", "x": 200, "y": 100}])])
    p = {"detections": [det(BOX, [(100, 105, 0.9, None), (200, 160, 0.9, None)])]}  # errors 0.01 and 0.12
    agg = aggregate([frame_records(f, p, 0.5, 0.25)])
    assert agg["pck@0.02"] == 0.5 and agg["pck@0.1"] == 0.5
    assert agg["frame_eye_hit_rate"] == 1.0 and agg["bird_recall"] == 1.0


def test_auroc_basics():
    assert auroc([(1, 0.9), (0, 0.1)]) == 1.0
    assert auroc([(1, 0.1), (0, 0.9)]) == 0.0
    assert auroc([(1, 0.5), (0, 0.5)]) == 0.5
    assert auroc([(1, 0.5)]) is None


def test_auroc_matches_pairwise_definition_with_ties():
    rng = random.Random(3)
    pairs = [(rng.randint(0, 1), round(rng.random(), 1)) for _ in range(300)]
    pos = [s for y, s in pairs if y]
    neg = [s for y, s in pairs if not y]
    brute = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert auroc(pairs) == pytest.approx(brute)


def test_choose_threshold_separates_visible_from_hidden():
    frames, preds = [], {}
    for i in range(10):
        fid = f"f{i}"
        frames.append(frame([bird(BOX, [{"state": "visible", "x": 100, "y": 100}, {"state": "hidden"}])], image_id=fid))
        # true eye at 0.8, spurious hidden-side eye at 0.3
        preds[fid] = {"detections": [det(BOX, [(100, 100, 0.8, None), (250, 300, 0.3, None)])]}
    t = choose_threshold(frames, preds, 0.25)
    assert 0.3 < t <= 0.8


def test_bootstrap_intervals_bracket_the_point_estimate():
    recs = []
    for i in range(40):
        f = frame([bird(BOX, [{"state": "visible", "x": 100, "y": 100}, {"state": "hidden"}])], image_id=f"f{i}")
        dy = 5 if i % 2 else 100  # half the eyes within 0.05, half not
        recs.append(frame_records(f, {"detections": [det(BOX, [(100, 100 + dy, 0.9, None)])]}, 0.5, 0.25))
    point = aggregate(recs)["pck@0.05"]
    lo, hi = bootstrap(recs, n=200)["pck@0.05"]
    assert lo <= point <= hi and lo < hi


def test_split_replay_matches_prepare_from_coco():
    species = [f"{i:03d}.S" for i in range(1, 201)]
    order_with_repeats = [s for s in species for _ in range(3)]
    val = eye_pose_v0_val_species(order_with_repeats)
    # the replay must equal the original recipe: seed the RNG, shuffle unique folders in order, take 15%
    random.seed(42)
    keys = list(dict.fromkeys(order_with_repeats))
    random.shuffle(keys)
    assert val == set(keys[:30]) and len(val) == 30


def test_agree_min_filters_only_detections_with_an_agreement_score():
    f = frame([], kind="negative")
    lone = det([0, 0, 10, 10], [(5, 5, 0.9, None)])
    agreed = {**det([20, 20, 30, 30], [(25, 25, 0.9, None)]), "agree": 0.6}
    weak = {**det([40, 40, 50, 50], [(45, 45, 0.9, None)]), "agree": 0.05}
    r = frame_records(f, {"detections": [lone, agreed, weak]}, 0.5, 0.25, agree_min=0.3)
    assert r["false_birds"] == 2  # the score-less detection and the agreed one survive; the weak one is dropped


def test_calibrate_prefers_the_setting_that_removes_false_eyes():
    from training.eval_eye_benchmark import calibrate

    frames, preds = [], {}
    for i in range(6):
        fid = f"b{i}"
        frames.append(frame([bird(BOX, [{"state": "visible", "x": 100, "y": 100}, {"state": "hidden"}])], image_id=fid))
        preds[fid] = {"detections": [{**det(BOX, [(100, 101, 0.9, None)]), "agree": 0.8}]}
    for i in range(6):
        fid = f"n{i}"
        frames.append(frame([], kind="negative", image_id=fid))
        preds[fid] = {"detections": [{**det([0, 0, 50, 50], [(5, 5, 0.9, None)], conf=0.9), "agree": 0.0}]}
    out = calibrate(frames, preds, [0.25, 0.5], [0.0, 0.3])
    best = out["best"]
    assert best["agree_min"] == 0.3 and best["negative_frames_with_false_eye"] == 0.0 and best["pck@0.05"] == 1.0


def test_crop_agreement_maps_rtmdet_boxes_back_to_the_frame(monkeypatch):
    import numpy as np
    from PIL import Image

    import training.teacher.pseudo_label_rtmdet as rt
    from training.eval_eye_benchmark import TopDownYoloAdapter

    calls = []

    def fake_run_teacher(model, rgb, device):
        calls.append(rgb.shape)
        # crop-pixel boxes: one bird matching the detector box, one bird elsewhere, one non-bird on top
        boxes = np.array([[10, 10, 50, 50], [60, 60, 70, 70], [10, 10, 50, 50]], dtype=float)
        return boxes, np.array([0.7, 0.9, 0.95]), np.array([rt.BIRD, rt.BIRD, 0])

    monkeypatch.setattr(rt, "run_teacher", fake_run_teacher)
    adapter = TopDownYoloAdapter.__new__(TopDownYoloAdapter)
    adapter.rtmdet, adapter.rtmdet_device = None, "cpu"
    im = Image.new("RGB", (200, 100))
    # detector box (110,30)-(150,70) inside crop (100,20)-(160,80): the matching RTMDet bird is at crop (10,10)-(50,50)
    score = adapter._crop_agreement(im, [110, 30, 150, 70], (100, 20, 160, 80))
    assert score == pytest.approx(0.7) and calls == [(60, 60, 3)]


def test_parse_model_spec():
    s = parse_model_spec("A=yolo,weights=models/x.pt,eyes=1:2")
    assert s == {"name": "A", "type": "yolo", "weights": "models/x.pt", "eyes": "1:2"}
    assert parse_model_spec("C=simcc,onnx=D:/m.onnx,box=gt")["onnx"] == "D:/m.onnx"
    with pytest.raises(ValueError):
        parse_model_spec("C=simcc,box=gt")
    with pytest.raises(ValueError):
        parse_model_spec("Z=nope")


def test_field_frames_rejects_hash_mismatch(tmp_path):
    (tmp_path / "manifest.jsonl").write_text(json.dumps({
        "image_id": "b0001", "kind": "bird", "image_path": "x.jpg", "width": 10, "height": 10,
        "rendition_sha256": "a" * 64, "size_stratum": "small", "body": "Z8"}) + "\n", encoding="utf-8")
    good = {"image_id": "b0001", "rendition_sha256": "a" * 64, "excluded": False, "birds": []}
    (tmp_path / "labels.jsonl").write_text(json.dumps(good) + "\n", encoding="utf-8")
    frames = field_frames(tmp_path)
    assert frames[0]["kind"] == "negative" and frames[0]["sampled_as"] == "bird"
    (tmp_path / "labels.jsonl").write_text(json.dumps({**good, "rendition_sha256": "b" * 64}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        field_frames(tmp_path)


def test_field_frames_birds_only(tmp_path):
    rows, labels, tags = [], [], []
    subject = {"id": "s", "bbox": [0, 0, 5, 5], "eyes": [{"state": "hidden"}, {"state": "hidden"}]}
    for i, kinds in enumerate([["bird", "other"], ["other"], []]):
        iid = f"b{i}"
        rows.append({"image_id": iid, "kind": "bird", "image_path": "x.jpg", "width": 10, "height": 10,
                     "rendition_sha256": "a" * 64})
        labels.append({"image_id": iid, "rendition_sha256": "a" * 64, "excluded": False, "birds": [subject] * len(kinds)})
        tags += [{"image_id": iid, "bird_index": k, "subject": s} for k, s in enumerate(kinds)]
    for name, data in (("manifest", rows), ("labels", labels), ("subjects", tags)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8")
    assert [(f["image_id"], f["kind"], len(f["birds"])) for f in field_frames(tmp_path)] == [
        ("b0", "bird", 2), ("b1", "bird", 1), ("b2", "negative", 0)]
    # mixed frame keeps its bird; an all-other frame is dropped; an empty frame stays a negative
    assert [(f["image_id"], f["kind"], len(f["birds"])) for f in field_frames(tmp_path, birds_only=True)] == [
        ("b0", "bird", 1), ("b2", "negative", 0)]
