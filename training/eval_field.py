"""Evaluate bird boxes and eye centers against reviewed field-image annotations.

Inputs are JSONL in one display-oriented rendition space. This module deliberately
does not decode images or run models: predictions and human labels can be frozen,
hashed, and compared without changing the evaluator between weight experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

SIDES = ("left", "right")
SPLITS = {"calibration", "test"}
EYE_STATES = {"visible", "hidden", "unresolvable", "unknown"}


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _hash(value: object, field: str) -> str:
    value = _text(value, field)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _bbox(value: object, width: int, height: int, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} must be [x1,y1,x2,y2]")
    x1, y1, x2, y2 = (_number(v, field) for v in value)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{field} must have positive area inside the display-oriented frame")
    return x1, y1, x2, y2


def _point(value: dict, width: int, height: int, field: str) -> None:
    x, y = _number(value.get("x"), f"{field}.x"), _number(value.get("y"), f"{field}.y")
    if not (0 <= x <= width and 0 <= y <= height):
        raise ValueError(f"{field} must lie inside the frame")


def validate_manifest_records(records: list[dict]) -> dict[str, dict]:
    """Require fully reviewed bird presence and prohibit group leakage."""
    if not records:
        raise ValueError("manifest is empty")
    by_id: dict[str, dict] = {}
    group_splits: dict[str, str] = {}
    rendition_splits: dict[str, str] = {}
    for item in records:
        if not isinstance(item, dict) or item.get("schema_version") != 1:
            raise ValueError("manifest schema_version must be 1")
        image_id = _text(item.get("image_id"), "image_id")
        if image_id in by_id:
            raise ValueError(f"duplicate image_id: {image_id}")
        _text(item.get("image_path"), f"{image_id}.image_path")
        group = _text(item.get("group_id"), f"{image_id}.group_id")
        split = item.get("split")
        if split not in SPLITS:
            raise ValueError(f"{image_id}.split must be calibration or test")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"group_id {group} crosses calibration and test")
        group_splits[group] = split
        if item.get("pixel_space") != "display_oriented":
            raise ValueError(f"{image_id}.pixel_space must be display_oriented")
        rendition_hash = _hash(item.get("rendition_sha256"), f"{image_id}.rendition_sha256")
        if rendition_hash in rendition_splits and rendition_splits[rendition_hash] != split:
            raise ValueError(f"rendition_sha256 {rendition_hash} crosses calibration and test")
        rendition_splits[rendition_hash] = split
        width, height = item.get("width"), item.get("height")
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError(f"{image_id} requires positive integer width and height")
        if item.get("reviewed") is not True:
            raise ValueError(f"{image_id} must be human-reviewed; empty birds means verified bird-free")
        birds = item.get("birds")
        if not isinstance(birds, list):
            raise ValueError(f"{image_id}.birds must be a list")
        bird_ids: set[str] = set()
        for bird in birds:
            if not isinstance(bird, dict):
                raise ValueError(f"{image_id}.birds must contain objects")
            bird_id = _text(bird.get("id"), f"{image_id}.bird.id")
            if bird_id in bird_ids:
                raise ValueError(f"{image_id} has duplicate bird id {bird_id}")
            bird_ids.add(bird_id)
            _bbox(bird.get("bbox"), width, height, f"{image_id}.{bird_id}.bbox")
            if "eyes" not in bird:
                continue  # box-only review: this frame is excluded from eye metrics
            eyes = bird["eyes"]
            if not isinstance(eyes, dict) or set(eyes) != set(SIDES):
                raise ValueError(f"{image_id}.{bird_id}.eyes must label both anatomical sides")
            for side in SIDES:
                eye = eyes[side]
                if not isinstance(eye, dict) or eye.get("state") not in EYE_STATES:
                    raise ValueError(f"{image_id}.{bird_id}.eyes.{side} has invalid state")
                if eye["state"] == "visible":
                    _point(eye, width, height, f"{image_id}.{bird_id}.eyes.{side}")
                elif "x" in eye or "y" in eye:
                    raise ValueError(f"{image_id}.{bird_id}.eyes.{side} has coordinates without a visible eye")
        by_id[image_id] = item
    return by_id


def _validate_predictions(records: list[dict], truth: dict[str, dict]) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("prediction rows must be objects")
        image_id = _text(item.get("image_id"), "prediction.image_id")
        if image_id not in truth or image_id in by_id:
            raise ValueError(f"unknown or duplicate prediction image_id: {image_id}")
        gt = truth[image_id]
        if item.get("rendition_sha256") != gt["rendition_sha256"]:
            raise ValueError(f"{image_id}.rendition_sha256 differs from the annotation rendition")
        detections = item.get("detections")
        if not isinstance(detections, list):
            raise ValueError(f"{image_id}.detections must be a list")
        for det in detections:
            if not isinstance(det, dict):
                raise ValueError(f"{image_id}.detections must contain objects")
            _bbox(det.get("bbox"), gt["width"], gt["height"], f"{image_id}.prediction.bbox")
            conf = _number(det.get("confidence"), f"{image_id}.prediction.confidence")
            if not 0 <= conf <= 1:
                raise ValueError(f"{image_id}.prediction.confidence must be in [0,1]")
            eyes = det.get("eyes", {})
            if not isinstance(eyes, dict) or not set(eyes).issubset(SIDES):
                raise ValueError(f"{image_id}.prediction.eyes has invalid sides")
            for side, eye in eyes.items():
                if not isinstance(eye, dict):
                    raise ValueError(f"{image_id}.prediction.eyes.{side} must be an object")
                _point(eye, gt["width"], gt["height"], f"{image_id}.prediction.eyes.{side}")
                eye_conf = _number(eye.get("confidence"), f"{image_id}.prediction.eyes.{side}.confidence")
                if not 0 <= eye_conf <= 1:
                    raise ValueError(f"{image_id}.prediction.eyes.{side}.confidence must be in [0,1]")
        by_id[image_id] = item
    if set(by_id) != set(truth):
        raise ValueError(f"prediction file is missing {len(set(truth) - set(by_id))} manifest images")
    return by_id


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def _match(detections: list[dict], birds: list[dict], threshold: float) -> dict[int, int]:
    """Maximum-cardinality one-to-one IoU matching, with stable IoU preference."""
    adjacency = [
        sorted((j for j, bird in enumerate(birds) if _iou(det["bbox"], bird["bbox"]) >= threshold),
               key=lambda j: (-_iou(det["bbox"], birds[j]["bbox"]), j))
        for det in detections
    ]
    owner: dict[int, int] = {}

    def assign(pred_idx: int, seen: set[int]) -> bool:
        for bird_idx in adjacency[pred_idx]:
            if bird_idx in seen:
                continue
            seen.add(bird_idx)
            if bird_idx not in owner or assign(owner[bird_idx], seen):
                owner[bird_idx] = pred_idx
                return True
        return False

    for pred_idx in sorted(range(len(detections)), key=lambda i: (-detections[i]["confidence"], i)):
        assign(pred_idx, set())
    return {pred_idx: bird_idx for bird_idx, pred_idx in owner.items()}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _p90(values: list[float]) -> float | None:
    return sorted(values)[math.ceil(0.9 * len(values)) - 1] if values else None


def evaluate_records(
    ground_truth: list[dict], predictions: list[dict], *, bird_conf: float = 0.25,
    eye_conf: float = 0.8, iou_threshold: float = 0.5, eye_tolerance_frac: float = 0.02,
) -> dict:
    """Return aggregate instance, frame, and visible-eye metrics with explicit denominators."""
    if not (0 <= bird_conf <= 1 and 0 <= eye_conf <= 1 and 0 < iou_threshold <= 1 and 0 < eye_tolerance_frac <= 1):
        raise ValueError("invalid evaluation thresholds")
    truth = validate_manifest_records(ground_truth)
    preds = _validate_predictions(predictions, truth)
    counts = {key: 0 for key in (
        "frames", "bird_positive_frames", "bird_frames_matched", "bird_free_frames",
        "bird_free_frames_with_fp", "bird_tp", "bird_fp", "bird_fn",
        "eye_evaluable_frames", "visible_eyes", "eye_tp", "eye_fp", "eye_fn",
    )}
    eye_errors_px: list[float] = []
    eye_errors_frac: list[float] = []
    for image_id, gt in truth.items():
        counts["frames"] += 1
        birds = gt["birds"]
        accepted = [det for det in preds[image_id]["detections"] if det["confidence"] >= bird_conf]
        matches = _match(accepted, birds, iou_threshold)
        counts["bird_tp"] += len(matches)
        counts["bird_fp"] += len(accepted) - len(matches)
        counts["bird_fn"] += len(birds) - len(matches)
        if birds:
            counts["bird_positive_frames"] += 1
            counts["bird_frames_matched"] += bool(matches)
        else:
            counts["bird_free_frames"] += 1
            counts["bird_free_frames_with_fp"] += bool(accepted)

        # Mixed box-only and eye-reviewed birds leave eye false positives ambiguous.
        if any("eyes" not in bird for bird in birds):
            continue
        counts["eye_evaluable_frames"] += 1
        for bird in birds:
            for side in SIDES:
                counts["visible_eyes"] += bird["eyes"][side]["state"] == "visible"
        for pred_idx, det in enumerate(accepted):
            bird_idx = matches.get(pred_idx)
            eyes = det.get("eyes", {})
            for side in SIDES:
                predicted = eyes.get(side)
                if predicted is not None and predicted["confidence"] < eye_conf:
                    predicted = None
                if bird_idx is None:
                    counts["eye_fp"] += predicted is not None
                    continue
                bird = birds[bird_idx]
                annotated = bird["eyes"][side]
                if annotated["state"] in {"unknown", "unresolvable"}:
                    continue
                if annotated["state"] == "hidden":
                    counts["eye_fp"] += predicted is not None
                    continue
                if predicted is None:
                    counts["eye_fn"] += 1
                    continue
                distance = math.hypot(predicted["x"] - annotated["x"], predicted["y"] - annotated["y"])
                box = bird["bbox"]
                diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
                eye_errors_px.append(distance)
                eye_errors_frac.append(distance / diagonal)
                if distance <= eye_tolerance_frac * diagonal:
                    counts["eye_tp"] += 1
                else:
                    counts["eye_fp"] += 1
                    counts["eye_fn"] += 1
        matched_birds = set(matches.values())
        for bird_idx, bird in enumerate(birds):
            if bird_idx not in matched_birds:
                counts["eye_fn"] += sum(bird["eyes"][side]["state"] == "visible" for side in SIDES)

    return {
        **counts,
        "bird_precision": _ratio(counts["bird_tp"], counts["bird_tp"] + counts["bird_fp"]),
        "bird_instance_recall": _ratio(counts["bird_tp"], counts["bird_tp"] + counts["bird_fn"]),
        "bird_frame_recall": _ratio(counts["bird_frames_matched"], counts["bird_positive_frames"]),
        "bird_free_frame_fp_rate": _ratio(counts["bird_free_frames_with_fp"], counts["bird_free_frames"]),
        "bird_fp_per_frame": counts["bird_fp"] / counts["frames"],
        "eye_precision": _ratio(counts["eye_tp"], counts["eye_tp"] + counts["eye_fp"]),
        "visible_eye_recall": _ratio(counts["eye_tp"], counts["visible_eyes"]),
        "eye_error_px_median": statistics.median(eye_errors_px) if eye_errors_px else None,
        "eye_error_px_p90": _p90(eye_errors_px),
        "eye_error_box_diagonal_frac_median": statistics.median(eye_errors_frac) if eye_errors_frac else None,
        "settings": {
            "bird_conf": bird_conf, "eye_conf": eye_conf, "iou_threshold": iou_threshold,
            "eye_tolerance_frac_of_gt_box_diagonal": eye_tolerance_frac,
        },
    }


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Human-reviewed JSONL manifest")
    parser.add_argument("predictions", type=Path, help="Model predictions JSONL on the same renditions")
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--bird-conf", type=float, default=0.25)
    parser.add_argument("--eye-conf", type=float, default=0.8)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--eye-tolerance-frac", type=float, default=0.02)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    all_truth = read_jsonl(args.manifest)
    validate_manifest_records(all_truth)  # leakage check must span both splits
    truth = [item for item in all_truth if item["split"] == args.split]
    if not truth:
        parser.error(f"manifest has no {args.split} images")
    ids = {item["image_id"] for item in truth}
    predictions = [item for item in read_jsonl(args.predictions) if item.get("image_id") in ids]
    report = evaluate_records(truth, predictions, bird_conf=args.bird_conf, eye_conf=args.eye_conf,
                              iou_threshold=args.iou, eye_tolerance_frac=args.eye_tolerance_frac)
    report["split"] = args.split
    report["manifest"] = str(args.manifest)
    report["predictions"] = str(args.predictions)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
