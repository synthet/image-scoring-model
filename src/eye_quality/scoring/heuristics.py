"""Classical sharpness and clarity metrics on eye crops."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from eye_quality.schemas import FailureType


def compute_eye_metrics(gray: np.ndarray, crop_width_px: int) -> dict[str, float | FailureType]:
    """Compute heuristic metrics for a single eye crop."""
    if gray.size == 0 or gray.shape[0] < 2 or gray.shape[1] < 2:
        return {
            "sharpness_score": 0.0,
            "clarity_score": 0.0,
            "edge_energy": 0.0,
            "noise_penalty": 1.0,
            "size_factor": 0.0,
            "failure_type": FailureType.HIDDEN_EYE,
        }

    arr = gray.astype(np.float64)
    lap_var = _laplacian_variance(arr)
    edge_energy = _tenengrad_energy(arr)
    clarity = _local_rms_contrast(arr)
    noise_penalty = _noise_penalty(arr)
    size_factor = _size_factor(crop_width_px)
    exposure_failure = _exposure_failure(arr)

    sharpness = _normalize_laplacian(lap_var)
    clarity_score = float(np.clip(clarity / 64.0, 0.0, 1.0))
    edge_norm = float(np.clip(edge_energy / 80.0, 0.0, 1.0))

    return {
        "sharpness_score": sharpness,
        "clarity_score": clarity_score,
        "edge_energy": edge_norm,
        "noise_penalty": noise_penalty,
        "size_factor": size_factor,
        "failure_type": exposure_failure,
        "laplacian_var": lap_var,
    }


def _as_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _laplacian_variance(arr: np.ndarray) -> float:
    gray = _as_uint8(arr)
    if cv2 is not None:
        try:
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            return float(lap.var())
        except cv2.error:
            pass
    center = gray[1:-1, 1:-1].astype(np.float64)
    lap = (
        -4.0 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(lap))


def _tenengrad_energy(arr: np.ndarray) -> float:
    gray = _as_uint8(arr)
    if cv2 is not None:
        try:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            return float(np.mean(mag))
        except cv2.error:
            pass
    gx = np.diff(gray.astype(np.float64), axis=1, prepend=gray[:, :1])
    gy = np.diff(gray.astype(np.float64), axis=0, prepend=gray[:1, :])
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag))


def _noise_penalty(arr: np.ndarray) -> float:
    gray = _as_uint8(arr)
    if cv2 is not None:
        try:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
            residual = np.abs(gray.astype(np.float64) - blurred.astype(np.float64))
            high_freq = float(np.mean(residual))
            return float(np.clip(high_freq / 25.0, 0.0, 1.0))
        except cv2.error:
            pass
    padded = np.pad(gray, 1, mode="edge")
    blurred = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0
    residual = np.abs(gray.astype(np.float64) - blurred)
    high_freq = float(np.mean(residual))
    return float(np.clip(high_freq / 25.0, 0.0, 1.0))


def _local_rms_contrast(arr: np.ndarray) -> float:
    mean = float(np.mean(arr))
    return float(np.sqrt(np.mean((arr - mean) ** 2)))


def _size_factor(crop_width_px: int, min_usable: int = 24) -> float:
    if crop_width_px <= 0:
        return 0.0
    if crop_width_px < min_usable:
        return float(crop_width_px / min_usable) * 0.5
    return float(np.clip(crop_width_px / 48.0, 0.5, 1.0))


def _normalize_laplacian(lap_var: float, cap: float = 500.0) -> float:
    """Higher variance => sharper => higher score."""
    return float(np.clip(lap_var / cap, 0.0, 1.0))


def _exposure_failure(arr: np.ndarray) -> FailureType:
    mean_brightness = float(np.mean(arr))
    highlight_ratio = float(np.mean(arr >= 250))
    shadow_ratio = float(np.mean(arr <= 5))
    if mean_brightness > 220 or highlight_ratio > 0.15:
        return FailureType.OVEREXPOSED
    if mean_brightness < 35 or shadow_ratio > 0.25:
        return FailureType.LOW_CONTRAST
    return FailureType.NONE
