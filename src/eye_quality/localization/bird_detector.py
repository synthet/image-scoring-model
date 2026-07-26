"""YOLO detect wrapper for wildlife bird bounding boxes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DETECT_WEIGHTS = _REPO_ROOT / "models" / "bird_detect_v0.pt"


def resolve_detect_weights_path(weights: str | Path | None = None) -> str:
    """Resolve detect weights; raise if the default checkpoint is missing.

    Unlike pose resolution, there is no COCO fallback — ``yolo11n.pt`` emits
    80 classes and would silently break the single-class ``bird`` contract.
    """
    if weights is not None and str(weights).strip():
        return str(weights)
    if DEFAULT_DETECT_WEIGHTS.is_file():
        return str(DEFAULT_DETECT_WEIGHTS)
    raise FileNotFoundError(
        f"Detect weights not found at {DEFAULT_DETECT_WEIGHTS}. "
        "Train with training/train_detect.py or place bird_detect_v0.pt under models/."
    )


@dataclass
class BirdBox:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    image_width: int = 0
    image_height: int = 0

    @property
    def bbox_norm(self) -> list[float]:
        """Normalized ``[x, y, w, h]`` in oriented image space."""
        if self.image_width <= 0 or self.image_height <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = self.bbox_xyxy
        return [
            x1 / self.image_width,
            y1 / self.image_height,
            max(0.0, x2 - x1) / self.image_width,
            max(0.0, y2 - y1) / self.image_height,
        ]

    @property
    def area_frac(self) -> float:
        if self.image_width <= 0 or self.image_height <= 0:
            return 0.0
        x1, y1, x2, y2 = self.bbox_xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return area / float(self.image_width * self.image_height)


class BirdDetector:
    """Run Ultralytics YOLO detect and return bird bounding boxes."""

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str | None = None,
        imgsz: int = 640,
    ) -> None:
        self.weights = resolve_detect_weights_path(weights)
        self.device = device
        self.imgsz = imgsz
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        logger.info("Loading YOLO detect weights: %s", self.weights)
        self._model = YOLO(self.weights)
        return self._model

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        max_det: int = 10,
    ) -> list[BirdBox]:
        """Run detect inference; returns boxes sorted by confidence descending."""
        from PIL import Image

        image_path = Path(image_path)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size

        model = self._load_model()
        kwargs: dict[str, Any] = {
            "imgsz": self.imgsz,
            "verbose": False,
            "conf": conf,
            "max_det": max_det,
        }
        if self.device:
            kwargs["device"] = self.device
        results = model.predict(str(image_path), **kwargs)

        boxes_out: list[BirdBox] = []
        if not results:
            return boxes_out

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return boxes_out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            boxes_out.append(
                BirdBox(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(confs[i]),
                    image_width=width,
                    image_height=height,
                )
            )

        boxes_out.sort(key=lambda b: b.confidence, reverse=True)
        return boxes_out
