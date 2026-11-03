"""Two-stage (detector crop -> pose) localization; fake models, so this runs in the fast subset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eye_quality.localization.keypoint_schema import KEYPOINT_NAMES
from eye_quality.localization.pose_model import PoseLocalizer
from eye_quality.pipeline import PipelineConfig


@dataclass
class FakeBox:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, image_path, conf=0.25, max_det=10):
        self.calls.append(conf)
        return list(self.boxes)


class _T:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _Kpts:
    def __init__(self, rows):
        self.data = [_T(r) for r in rows]


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _T(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4))
        self.conf = _T(conf)


class _Result:
    def __init__(self, rows, boxes=None, confs=None):
        self.keypoints = _Kpts(rows)
        self.boxes = _Boxes(boxes, confs) if boxes is not None else None


class FakePose:
    """Returns one keypoint row per crop: every point at (crop_w*0.25, crop_h*0.5) except the eyes."""

    def __init__(self, eyes_in_crop):
        self.eyes_in_crop = eyes_in_crop
        self.crops = []

    def predict(self, source, **kwargs):
        assert isinstance(source, Image.Image), "two-stage must pass the crop, not the path"
        self.crops.append(source.size)
        row = np.zeros((len(KEYPOINT_NAMES), 3))
        row[:, 2] = 0.9
        for name, (x, y) in self.eyes_in_crop.items():
            row[KEYPOINT_NAMES.index(name)] = [x, y, 0.95]
        w, h = source.size
        return [_Result([row], boxes=[(0, 0, w, h)], confs=[0.9])]


def _row(eyes):
    row = np.zeros((len(KEYPOINT_NAMES), 3))
    row[:, 2] = 0.9
    for name, (x, y) in eyes.items():
        row[KEYPOINT_NAMES.index(name)] = [x, y, 0.95]
    return row


class ScriptedPose:
    """One scripted answer per crop: ``None`` for no pose detection, else a list of (box_in_crop, conf, eyes)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.crops = []

    def predict(self, source, **kwargs):
        self.crops.append(source.size)
        answer = self.answers.pop(0)
        if not answer:
            return []
        return [_Result([_row(eyes) for _, _, eyes in answer],
                        boxes=[b for b, _, _ in answer], confs=[c for _, c, _ in answer])]


def _image(tmp_path: Path, w=200, h=100) -> Path:
    p = tmp_path / "frame.jpg"
    Image.new("RGB", (w, h), (90, 90, 90)).save(p)
    return p


def _localizer(detector, pose, **kwargs) -> PoseLocalizer:
    loc = PoseLocalizer(weights="unused.pt", crop_detector=detector, crop_pad=0.25, **kwargs)
    loc._model = pose
    return loc


def test_crop_keypoints_are_mapped_back_to_frame(tmp_path):
    # box 40x40 at (50,20); 25% padding -> crop (40,10)-(100,70)
    det = FakeDetector([FakeBox((50, 20, 90, 60), 0.9)])
    pose = FakePose({"left_eye": (10, 20), "right_eye": (30, 20)})
    out = _localizer(det, pose).predict(_image(tmp_path))
    assert pose.crops == [(60, 60)]
    kp = {k.name: k for k in out[0].keypoints}
    assert (kp["left_eye"].x, kp["left_eye"].y) == pytest.approx((50, 30))
    assert (kp["right_eye"].x, kp["right_eye"].y) == pytest.approx((70, 30))
    assert out[0].bbox_xyxy == (50, 20, 90, 60) and out[0].bbox_confidence == pytest.approx(0.9)
    assert det.calls == [0.25]  # detector gated at the pose bbox threshold


def test_crop_is_clamped_at_frame_edges_and_tiny_boxes_skipped(tmp_path):
    det = FakeDetector([FakeBox((0, 0, 30, 30), 0.8), FakeBox((150, 50, 153, 53), 0.95)])
    pose = FakePose({"left_eye": (5, 5)})
    out = _localizer(det, pose).predict(_image(tmp_path))
    assert pose.crops == [(38, 38)]  # (0,0)-(38,38); the 3 px box gives a crop under 8 px and is skipped
    assert len(out) == 1
    left = next(k for k in out[0].keypoints if k.name == "left_eye")
    assert (left.x, left.y) == pytest.approx((5, 5))


def test_detections_sorted_by_detector_confidence(tmp_path):
    det = FakeDetector([FakeBox((10, 10, 60, 60), 0.4), FakeBox((100, 20, 160, 80), 0.9)])
    out = _localizer(det, FakePose({})).predict(_image(tmp_path))
    assert [d.bbox_confidence for d in out] == pytest.approx([0.9, 0.4])


def test_localize_eyes_uses_the_crop_path(tmp_path):
    det = FakeDetector([FakeBox((50, 20, 150, 90), 0.9)])
    pose = FakePose({"left_eye": (40, 30), "right_eye": (90, 30)})
    locs, _, notes = _localizer(det, pose).localize_eyes(_image(tmp_path), 200, 100)
    assert "no_pose_detection" not in notes
    centres = sorted((round(e.center_x), round(e.center_y)) for e in locs)
    # padding 25x17.5 -> crop origin (25, round(2.5) == 2), so eyes land at (65, 32) and (115, 32)
    assert centres == [(65, 32), (115, 32)]


def test_default_localizer_never_uses_crops(tmp_path, monkeypatch):
    loc = PoseLocalizer(weights="unused.pt")
    assert loc.crop_detector is None

    def boom(*_a, **_k):
        raise AssertionError("crop path used without a detector")

    monkeypatch.setattr(loc, "_predict_on_crops", boom)

    class EmptyModel:
        def predict(self, source, **kwargs):
            assert isinstance(source, str)
            return []

    loc._model = EmptyModel()
    assert loc.predict(_image(tmp_path)) == []


def test_pipeline_config_builds_two_stage_only_when_asked():
    assert PipelineConfig().make_localizer().crop_detector is None
    loc = PipelineConfig(two_stage=True, detect_weights="custom_detect.pt").make_localizer()
    assert loc.crop_detector is not None and loc.crop_detector.weights == "custom_detect.pt"


def test_cli_flags_parse():
    from eye_quality.cli import _build_parser

    args = _build_parser().parse_args(["score", "x.jpg", "--two-stage", "--detect-weights", "d.pt"])
    assert args.two_stage and args.detect_weights == "d.pt"
    assert _build_parser().parse_args(["batch", "m.jsonl", "--output", "o.jsonl"]).two_stage is False


def test_top_box_without_keypoints_does_not_shadow_bird(tmp_path):
    # The more confident box (a false positive) gets no pose; the bird is in the second box.
    det = FakeDetector([FakeBox((10, 10, 50, 50), 0.9), FakeBox((100, 20, 160, 80), 0.6)])
    # second crop: (85, 5)-(175, 95), target box in crop pixels (15, 15)-(75, 75)
    pose = ScriptedPose([None, [((15, 15, 75, 75), 0.8, {"left_eye": (30, 30), "right_eye": (60, 30)})]])
    locs, _, notes = _localizer(det, pose).localize_eyes(_image(tmp_path), 200, 100)
    assert "no_visible_eyes" not in notes
    assert sorted((round(e.center_x), round(e.center_y)) for e in locs) == [(115, 35), (145, 35)]


def test_without_any_eyes_the_top_box_is_still_reported(tmp_path):
    det = FakeDetector([FakeBox((10, 10, 50, 50), 0.9), FakeBox((100, 20, 160, 80), 0.6)])
    locs, _, notes = _localizer(det, ScriptedPose([None, None])).localize_eyes(_image(tmp_path), 200, 100)
    assert locs == [] and "no_visible_eyes" in notes


def test_detector_threshold_is_its_own_setting(tmp_path):
    det = FakeDetector([])
    _localizer(det, FakePose({}), crop_conf=0.4).predict(_image(tmp_path))
    assert det.calls == [0.4]
    assert PipelineConfig().make_localizer().crop_conf is None
    loc = PipelineConfig(two_stage=True, detect_weights="custom_detect.pt", detect_conf=0.4).make_localizer()
    assert loc.crop_conf == pytest.approx(0.4)


def test_neighbour_bird_in_padded_crop_is_not_used(tmp_path):
    # crop (40, 10)-(100, 70); target box in crop pixels (10, 10)-(50, 50).
    det = FakeDetector([FakeBox((50, 20, 90, 60), 0.9)])
    neighbour = ((52, 0, 60, 60), 0.95, {"left_eye": (55, 5)})  # sits in the padding only
    bird = ((12, 12, 48, 48), 0.7, {"left_eye": (20, 20), "right_eye": (40, 20)})
    out = _localizer(det, ScriptedPose([[neighbour, bird]])).predict(_image(tmp_path))
    kp = {k.name: k for k in out[0].keypoints}
    assert (kp["left_eye"].x, kp["left_eye"].y) == pytest.approx((60, 30))


def test_crop_with_only_a_neighbour_gets_no_keypoints(tmp_path):
    det = FakeDetector([FakeBox((50, 20, 90, 60), 0.9)])
    neighbour = ((52, 0, 60, 60), 0.95, {"left_eye": (55, 5)})
    out = _localizer(det, ScriptedPose([[neighbour]])).predict(_image(tmp_path))
    assert out[0].keypoints == []


def test_single_stage_selection_is_unchanged(tmp_path, monkeypatch):
    from eye_quality.localization.pose_model import PoseDetection

    no_eyes = PoseDetection(bbox_xyxy=(0, 0, 50, 50), bbox_confidence=0.9, image_width=200, image_height=100)
    with_eyes = PoseDetection(bbox_xyxy=(100, 0, 150, 50), bbox_confidence=0.5, image_width=200, image_height=100)
    with_eyes.keypoints = PoseLocalizer(weights="unused.pt")._parse_keypoints(_row({"left_eye": (120, 20)}))
    loc = PoseLocalizer(weights="unused.pt")
    monkeypatch.setattr(loc, "predict", lambda _p: [no_eyes, with_eyes])
    locs, _, notes = loc.localize_eyes(_image(tmp_path), 200, 100)
    assert locs == [] and "no_visible_eyes" in notes
