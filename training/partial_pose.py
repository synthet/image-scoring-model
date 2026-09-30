"""Partial-keypoint objective for already matched pose instances.

This primitive is not a stock Ultralytics trainer override. A caller must carry
``annotation_known`` through every transform and assignment before using it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import KeypointLoss


def masked_keypoint_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    annotation_known: torch.Tensor,
    area: torch.Tensor,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return location/objectness losses, with zero gradients for unknowns.

    ``pred`` and ``target`` are (instances, points, 3). Prediction channel 2 is
    an objectness logit; target channel 2 is COCO visibility (0/1/2). Coordinates
    and box areas must share a coordinate system. The known mask is Boolean.
    """
    if pred.ndim != 3 or pred.shape[-1] != 3 or target.shape != pred.shape:
        raise ValueError("pred and target must have identical (N, K, 3) shape")
    if annotation_known.dtype != torch.bool or annotation_known.shape != pred.shape[:2]:
        raise ValueError("annotation_known must be a boolean (N, K) mask")
    if pred.shape[1] == 0 or sigmas.shape != (pred.shape[1],) or area.shape != (pred.shape[0], 1):
        raise ValueError("Expected K sigmas and (N, 1) areas")
    if any(t.device != pred.device for t in (target, annotation_known, area, sigmas)):
        raise ValueError("All tensors must be on the prediction device")
    if not torch.isfinite(pred).all() or not torch.isfinite(area).all() or not (area > 0).all():
        raise ValueError("Predictions must be finite and areas finite and positive")
    if not torch.isfinite(sigmas).all() or not (sigmas > 0).all():
        raise ValueError("Sigmas must be finite and positive")
    visibility = target[..., 2]
    if not ((visibility[annotation_known] == 0) | (visibility[annotation_known] == 1) | (visibility[annotation_known] == 2)).all():
        raise ValueError("Known visibility values must be 0, 1, or 2")
    visible = annotation_known & (visibility > 0)
    if not torch.isfinite(target[..., :2][visible]).all():
        raise ValueError("Observed coordinates must be finite")
    zero = pred.sum() * 0.0
    if not annotation_known.any():
        return zero, zero

    # Sanitize unknown/absent coordinates before arithmetic: NaN * 0 is not 0.
    safe_pred = torch.where(visible[..., None], pred[..., :2], torch.zeros_like(pred[..., :2]))
    safe_target = torch.where(visible[..., None], target[..., :2], torch.zeros_like(target[..., :2]))
    # Box-only instances must not dilute location loss; fully labelled inputs
    # retain upstream's reduction, including known-all-absent instances.
    rows = annotation_known.any(dim=1)
    location = KeypointLoss(sigmas)(safe_pred[rows], safe_target[rows], visible[rows], area[rows])
    objectness = F.binary_cross_entropy_with_logits(
        pred[..., 2][annotation_known], visible[annotation_known].to(pred.dtype)
    )
    return location, objectness
