"""CLI tests for eye-quality detect subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from eye_quality.cli import main
from eye_quality.localization.bird_detector import BirdBox


def test_detect_cli_json_stdout(tmp_path: Path, monkeypatch, capsys):
    img = tmp_path / "bird.jpg"
    Image.new("RGB", (200, 100), color=(10, 20, 30)).save(img)
    weights = tmp_path / "bird_detect_v0.pt"
    weights.write_bytes(b"fake")

    def fake_predict(self, image_path, conf=0.25, max_det=10):
        return [
            BirdBox(
                bbox_xyxy=(20.0, 10.0, 180.0, 90.0),
                confidence=0.97,
                image_width=200,
                image_height=100,
            )
        ]

    monkeypatch.setattr(
        "eye_quality.localization.bird_detector.BirdDetector.predict",
        fake_predict,
    )
    monkeypatch.setattr(
        "eye_quality.localization.bird_detector.BirdDetector._load_model",
        lambda self: None,
    )

    code = main(
        [
            "detect",
            str(img),
            "--weights",
            str(weights),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "image" in payload
    assert "model" in payload
    assert "birds" in payload
    assert payload["image"]["width"] == 200
    assert payload["image"]["height"] == 100
    assert len(payload["birds"]) == 1
    bird = payload["birds"][0]
    assert bird["bbox_xyxy"] == [20.0, 10.0, 180.0, 90.0]
    assert bird["bbox_norm"] == [0.1, 0.1, 0.8, 0.8]
    assert bird["confidence"] == pytest.approx(0.97)
    assert bird["area_frac"] == pytest.approx(0.64)


def test_detect_cli_writes_output_file(tmp_path: Path, monkeypatch):
    img = tmp_path / "bird.jpg"
    Image.new("RGB", (100, 100), color=(1, 2, 3)).save(img)
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    out = tmp_path / "out.json"

    monkeypatch.setattr(
        "eye_quality.localization.bird_detector.BirdDetector.predict",
        lambda self, image_path, conf=0.25, max_det=10: [],
    )
    monkeypatch.setattr(
        "eye_quality.localization.bird_detector.BirdDetector._load_model",
        lambda self: None,
    )

    code = main(["detect", str(img), "--weights", str(weights), "--output", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["birds"] == []


def test_detect_cli_missing_file(tmp_path: Path):
    code = main(["detect", str(tmp_path / "missing.jpg")])
    assert code == 1
