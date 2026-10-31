"""Optional, local-only SAM 2.1 subject segmentation for diagnostic overlays."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROVIDER_VERSION = "sam2.1-tiny-box-v1"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


@lru_cache(maxsize=1)
def _predictor(checkpoint, modified, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(MODEL_CONFIG, checkpoint, device=device, apply_postprocessing=True))


def segment_subject(image, region, *, checkpoint, device="auto"):
    """A verified display-normalized primary box prompts a full-raster binary mask.

    No downloads, fallback rectangle, or scoring side effects. Confidence is the
    provider's predicted mask quality, not an image-quality score.
    """
    if len(region) != 4 or not all(np.isfinite(value) for value in region):
        raise ValueError("A finite normalized subject box is required")
    x0, y0, x1, y1 = region
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("Subject box is outside the display raster")
    weights = Path(checkpoint)
    if not weights.is_file():
        return {"state": "unavailable", "reason": "Local segmentation weights are not installed"}
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = _predictor(str(weights.resolve()), weights.stat().st_mtime_ns, device)
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    box = np.array([x0 * width, y0 * height, x1 * width, y1 * height], dtype=np.float32)
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    candidates = []
    for mask, score in zip(masks, scores):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (height, width) or not np.isfinite(score) or score < .5:
            continue
        area = int(mask.sum())
        if not area or area >= width * height * .95:
            continue
        inside = int(mask[int(y0 * height):max(int(y0 * height) + 1, int(y1 * height)), int(x0 * width):max(int(x0 * width) + 1, int(x1 * width))].sum())
        if inside / area < .5:
            continue
        candidates.append((float(score), mask))
    if not candidates:
        return {"state": "unavailable", "reason": "Subject segmentation confidence is insufficient"}
    score, mask = max(candidates, key=lambda value: value[0])
    hasher = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return {"mask": mask, "confidence": score, "provider_version": PROVIDER_VERSION + ":" + hasher.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description="Generate one local diagnostic subject mask")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--box", required=True, help="Four normalized JSON coordinates")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    arguments = parser.parse_args()
    with Image.open(arguments.input) as image:
        result = segment_subject(image, json.loads(arguments.box), checkpoint=arguments.checkpoint, device=arguments.device)
    if "mask" in result:
        Image.fromarray((result.pop("mask") * 255).astype(np.uint8)).save(arguments.output, format="PNG")
        result["state"] = "ready"
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
