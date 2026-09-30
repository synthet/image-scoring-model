"""Decode bird head/eye SimCC ONNX outputs for the Gate 0 teacher adapter.

Loads a local ONNX file by path only; weights are never committed. Training-use
permission and third-party lineage are a human gate (see docs/PRIVATE_LOCAL.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

POSE = 256
IMNET_MEAN = np.array([123.675 / 255, 116.28 / 255, 103.53 / 255], dtype=np.float64)
IMNET_STD = np.array([58.395 / 255, 57.12 / 255, 57.375 / 255], dtype=np.float64)
VIS_CALIB = (0.6343, 1.1351)
TEACHER_POINT_NAMES = ("eye0", "eye1", "beak")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def preprocess_crop(rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> 1x3x256x256 float32, ImageNet norm (scale /255 first)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HWC RGB uint8/float image")
    x = rgb.astype(np.float64)
    x = x / 255.0
    x = (x - IMNET_MEAN) / IMNET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None].astype(np.float32))


@dataclass(frozen=True)
class SimCCDecode:
    """Three teacher points in 256x256 input pixels plus raw visibility logits."""

    points: dict[str, tuple[float, float, float]]
    orientation_logits: np.ndarray | None

    @property
    def orientation_argmax(self) -> int | None:
        if self.orientation_logits is None or self.orientation_logits.size < 5:
            return None
        logits = self.orientation_logits[:5]
        return int(np.argmax(logits - logits.max()))


def decode_simcc_outputs(outputs: Mapping[str, np.ndarray]) -> SimCCDecode:
    """Map ONNX output tensors to eye0/eye1/beak in input-pixel space."""
    sx = np.asarray(outputs["simcc_x"][0], dtype=np.float64)
    sy = np.asarray(outputs["simcc_y"][0], dtype=np.float64)
    vis = outputs.get("kpt_visibility")
    orient = outputs.get("orientation")
    if sx.ndim != 2 or sy.ndim != 2 or sx.shape[0] < 3 or sy.shape[0] < 3:
        raise ValueError("simcc_x and simcc_y must be (K, bins) with K >= 3")

    def kp(k: int) -> tuple[float, float, float]:
        ix, iy = int(np.argmax(sx[k])), int(np.argmax(sy[k]))
        if vis is None:
            conf = min(1.0, max(0.0, (float(sx[k, ix]) + float(sy[k, iy])) / 2))
        else:
            v = np.asarray(vis, dtype=np.float64)
            conf = _sigmoid(VIS_CALIB[0] * float(v.reshape(-1)[k]) + VIS_CALIB[1])
        return ix / 2.0, iy / 2.0, conf

    points = {name: kp(i) for i, name in enumerate(TEACHER_POINT_NAMES)}
    orientation = None
    if orient is not None:
        orientation = np.asarray(orient, dtype=np.float64).reshape(-1)
    return SimCCDecode(points=points, orientation_logits=orientation)


class BirdEyeOnnxSession:
    """Thin onnxruntime wrapper; optional dependency."""

    def __init__(self, onnx_path: Path | str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for BirdEyeOnnxSession") from exc
        path = Path(onnx_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.path = path
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def run(self, rgb_crop: np.ndarray) -> SimCCDecode:
        feed = {self._input_name: preprocess_crop(rgb_crop)}
        raw = dict(zip(self._output_names, self._session.run(None, feed)))
        return decode_simcc_outputs(raw)
