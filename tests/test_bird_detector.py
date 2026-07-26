"""Unit tests for BirdDetector (no GPU / weights required)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from eye_quality.localization.bird_detector import (
    BirdBox,
    BirdDetector,
    resolve_detect_weights_path,
)


def test_bird_box_bbox_norm_and_area_frac():
    box = BirdBox(
        bbox_xyxy=(100.0, 50.0, 300.0, 250.0),
        confidence=0.9,
        image_width=1000,
        image_height=500,
    )
    assert box.bbox_norm == pytest.approx([0.1, 0.1, 0.2, 0.4])
    assert box.area_frac == pytest.approx(0.08)


def test_bird_box_area_frac_zero_when_size_unknown():
    box = BirdBox(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), confidence=0.5)
    assert box.area_frac == 0.0


def test_resolve_detect_weights_path_raises_when_missing(tmp_path: Path, monkeypatch):
    missing = tmp_path / "nope.pt"
    monkeypatch.setattr(
        "eye_quality.localization.bird_detector.DEFAULT_DETECT_WEIGHTS",
        missing,
    )
    with pytest.raises(FileNotFoundError, match="bird_detect"):
        resolve_detect_weights_path(None)


def test_resolve_detect_weights_path_explicit():
    assert resolve_detect_weights_path("models/custom.pt") == "models/custom.pt"


def test_predict_parses_ultralytics_boxes(tmp_path: Path, monkeypatch):
    img = tmp_path / "bird.jpg"
    Image.new("RGB", (200, 100), color=(40, 80, 120)).save(img)

    xyxy = np.array([[20.0, 10.0, 180.0, 90.0], [5.0, 5.0, 40.0, 40.0]], dtype=np.float32)
    confs = np.array([0.95, 0.40], dtype=np.float32)

    fake_boxes = SimpleNamespace(
        xyxy=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: xyxy)),
        conf=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: confs)),
    )
    fake_result = SimpleNamespace(boxes=fake_boxes)
    fake_model = MagicMock()
    fake_model.predict.return_value = [fake_result]

    weights = tmp_path / "bird_detect_v0.pt"
    weights.write_bytes(b"fake")
    detector = BirdDetector(weights=weights, device=None, imgsz=640)
    detector._model = fake_model

    boxes = detector.predict(img, conf=0.25)
    assert len(boxes) == 2
    assert boxes[0].confidence == pytest.approx(0.95)
    assert boxes[0].bbox_xyxy == (20.0, 10.0, 180.0, 90.0)
    assert boxes[0].image_width == 200
    assert boxes[0].image_height == 100
    assert boxes[0].bbox_norm == pytest.approx([0.1, 0.1, 0.8, 0.8])
    # Sorted by confidence descending
    assert boxes[0].confidence >= boxes[1].confidence
    fake_model.predict.assert_called_once()
