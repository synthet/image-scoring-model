"""Optional GPU inference tests (require models/eye_pose_v0.pt)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "models" / "eye_pose_v0.pt"
VAL_DIR = REPO_ROOT / "data" / "wildlife_bird" / "images" / "val"


@pytest.mark.gpu
def test_pose_inference_on_val_image():
    if not WEIGHTS.is_file():
        pytest.skip(f"weights not found: {WEIGHTS}")
    val_images = sorted(VAL_DIR.glob("*.jpg"))
    if not val_images:
        pytest.skip("no val images in data/wildlife_bird/images/val")

    from eye_quality import score_image
    from eye_quality.pipeline import PipelineConfig

    config = PipelineConfig(weights=str(WEIGHTS), device="0")
    result = score_image(val_images[0], config=config)
    assert result.eye_quality.model_name == "eye_quality_heuristic"
    assert 0.0 <= result.eye_quality.confidence <= 1.0
