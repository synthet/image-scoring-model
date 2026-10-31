import numpy as np
from PIL import Image
import pytest

from eye_quality import segmentation


def test_box_prompt_uses_pixels_and_preserves_provider_mask(monkeypatch, tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"test-weights")
    mask = np.zeros((48, 64), bool)
    mask[10:35, 15:45] = True
    observed = {}
    class Predictor:
        def set_image(self, image):
            observed["shape"] = image.shape
        def predict(self, **kwargs):
            observed.update(kwargs)
            return np.array([mask]), np.array([.95]), None
    monkeypatch.setattr(segmentation, "_predictor", lambda *args: Predictor())
    result = segmentation.segment_subject(Image.new("RGB", (64, 48)), (.2, .2, .8, .8), checkpoint=str(checkpoint), device="cpu")
    assert observed["shape"] == (48, 64, 3)
    assert np.allclose(observed["box"], [12.8, 9.6, 51.2, 38.4])
    assert np.array_equal(result["mask"], mask)
    assert result["confidence"] == .95
    assert result["provider_version"]


def test_missing_weights_never_downloads_or_imports_predictor(monkeypatch):
    monkeypatch.setattr(segmentation, "_predictor", lambda *args: pytest.fail("must not load"))
    result = segmentation.segment_subject(Image.new("RGB", (32, 32)), (.1, .1, .9, .9), checkpoint="not-installed.pt")
    assert result["state"] == "unavailable"


@pytest.mark.parametrize("box", [(-.1, 0, 1, 1), (0, 0, 0, 1), (0, 1, 1, .5)])
def test_invalid_prompt_is_rejected_without_model(box):
    with pytest.raises(ValueError):
        segmentation.segment_subject(Image.new("RGB", (32, 32)), box, checkpoint="not-installed.pt")
