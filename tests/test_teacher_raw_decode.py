"""Regression tests for RAW inputs to the shared teacher decoder."""

from io import BytesIO
from pathlib import Path

from PIL import Image

from training.teacher.pseudo_label_rtmdet import decode


def test_raw_decoder_prefers_full_embedded_jpeg_over_pillow_thumbnail(monkeypatch, tmp_path):
    raw_path = tmp_path / "bird.NEF"
    raw_path.write_bytes(b"placeholder")
    full = BytesIO()
    Image.new("RGB", (640, 480), "green").save(full, format="JPEG")
    monkeypatch.setattr(
        "training.build_eye_dataset.extract_full_preview",
        lambda path: (full.getvalue(), "subifd"),
    )
    original_open = Image.open

    def open_raw_as_thumbnail(source, *args, **kwargs):
        if isinstance(source, (str, Path)):
            return Image.new("RGB", (160, 120), "red")
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(Image, "open", open_raw_as_thumbnail)

    decoded = decode(str(raw_path))

    assert decoded.size == (640, 480)


def test_raw_decoder_uses_pillow_when_no_embedded_jpeg(monkeypatch, tmp_path):
    raw_path = tmp_path / "bird.NEF"
    raw_path.write_bytes(b"placeholder")
    monkeypatch.setattr("training.build_eye_dataset.extract_full_preview", lambda path: None)
    monkeypatch.setattr(Image, "open", lambda source: Image.new("RGB", (160, 120)))

    assert decode(str(raw_path)).size == (160, 120)
