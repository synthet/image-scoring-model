"""Ground-truth field evaluation, without model weights or image decoding."""

from __future__ import annotations

import unittest
import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

from training.eval_field import evaluate_records, validate_manifest_records


def rendition_hash(image_id: str) -> str:
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest()


def frame(image_id: str, birds: list[dict], *, group: str = "burst-1", split: str = "test") -> dict:
    return {
        "schema_version": 1,
        "image_id": image_id,
        "image_path": f"/private/{image_id}.jpg",
        "group_id": group,
        "split": split,
        "pixel_space": "display_oriented",
        "rendition_sha256": rendition_hash(image_id),
        "width": 100,
        "height": 100,
        "reviewed": True,
        "birds": birds,
    }


def bird(bird_id: str, bbox: list[int], eyes: dict | None = None) -> dict:
    result = {"id": bird_id, "bbox": bbox}
    if eyes is not None:
        result["eyes"] = eyes
    return result


def prediction(image_id: str, detections: list[dict]) -> dict:
    return {"image_id": image_id, "rendition_sha256": rendition_hash(image_id), "detections": detections}


def detection(bbox: list[int], eyes: dict | None = None) -> dict:
    return {"bbox": bbox, "confidence": 0.9, "eyes": eyes or {}}


class ManifestValidationTests(unittest.TestCase):
    def test_rejects_unreviewed_empty_frame(self) -> None:
        item = frame("empty", [])
        item["reviewed"] = False
        with self.assertRaisesRegex(ValueError, "reviewed"):
            validate_manifest_records([item])

    def test_rejects_group_crossing_calibration_and_test(self) -> None:
        records = [frame("a", [], split="calibration"), frame("b", [], split="test")]
        with self.assertRaisesRegex(ValueError, "group_id"):
            validate_manifest_records(records)

    def test_rejects_same_rendition_in_both_splits(self) -> None:
        records = [frame("a", [], group="g1", split="calibration"),
                   frame("b", [], group="g2", split="test")]
        records[1]["rendition_sha256"] = records[0]["rendition_sha256"]
        with self.assertRaisesRegex(ValueError, "rendition_sha256"):
            validate_manifest_records(records)

    def test_rejects_incomplete_eye_review(self) -> None:
        records = [frame("a", [bird("b1", [10, 10, 50, 50], {"left": {"state": "visible", "x": 20, "y": 20}})])]
        with self.assertRaisesRegex(ValueError, "eyes"):
            validate_manifest_records(records)


class FieldMetricsTests(unittest.TestCase):
    def test_cli_scores_selected_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            predictions = Path(directory) / "predictions.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in [
                frame("cal", [], group="g1", split="calibration"),
                frame("test", [bird("b1", [10, 10, 30, 30])], group="g2"),
            ]) + "\n", encoding="utf-8")
            predictions.write_text("\n".join(json.dumps(row) for row in [
                prediction("cal", []), prediction("test", [detection([10, 10, 30, 30])]),
            ]) + "\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "training.eval_field", str(manifest), str(predictions),
                 "--split", "test"],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
            )
            report = json.loads(proc.stdout)
            self.assertEqual((report["frames"], report["bird_tp"], report["bird_free_frames"]), (1, 1, 0))

    def test_matching_maximizes_recalled_birds(self) -> None:
        truth = [frame("a", [bird("b1", [0, 0, 20, 20]), bird("b2", [10, 0, 30, 20])])]
        preds = [prediction("a", [detection([5, 0, 25, 20]), detection([0, 0, 20, 20])])]
        result = evaluate_records(truth, preds, iou_threshold=0.5)
        self.assertEqual((result["bird_tp"], result["bird_fp"], result["bird_fn"]), (2, 0, 0))

    def test_duplicate_detection_is_false_positive_and_second_bird_is_missed(self) -> None:
        truth = [frame("a", [bird("b1", [0, 0, 20, 20]), bird("b2", [60, 60, 90, 90])])]
        preds = [prediction("a", [detection([0, 0, 20, 20]), detection([0, 0, 20, 20])])]
        result = evaluate_records(truth, preds)
        self.assertEqual((result["bird_tp"], result["bird_fp"], result["bird_fn"]), (1, 1, 1))
        self.assertEqual(result["bird_frame_recall"], 1.0)

    def test_verified_empty_frame_counts_false_positive(self) -> None:
        result = evaluate_records([frame("empty", [])], [prediction("empty", [detection([0, 0, 20, 20])])])
        self.assertEqual(result["bird_free_frames"], 1)
        self.assertEqual(result["bird_free_frames_with_fp"], 1)
        self.assertEqual(result["bird_free_frame_fp_rate"], 1.0)

    def test_upstream_bird_miss_counts_visible_eye_false_negative(self) -> None:
        eyes = {"left": {"state": "visible", "x": 20, "y": 20}, "right": {"state": "hidden"}}
        result = evaluate_records([frame("a", [bird("b1", [10, 10, 50, 50], eyes)])], [prediction("a", [])])
        self.assertEqual((result["eye_tp"], result["eye_fp"], result["eye_fn"]), (0, 0, 1))

    def test_hidden_eye_prediction_is_false_positive(self) -> None:
        eyes = {"left": {"state": "visible", "x": 20, "y": 20}, "right": {"state": "hidden"}}
        predicted_eyes = {"left": {"x": 20, "y": 20, "confidence": 0.9}, "right": {"x": 30, "y": 20, "confidence": 0.9}}
        result = evaluate_records([frame("a", [bird("b1", [10, 10, 50, 50], eyes)])], [prediction("a", [detection([10, 10, 50, 50], predicted_eyes)])])
        self.assertEqual((result["eye_tp"], result["eye_fp"], result["eye_fn"]), (1, 1, 0))

    def test_out_of_tolerance_eye_counts_false_positive_and_false_negative(self) -> None:
        eyes = {"left": {"state": "visible", "x": 20, "y": 20}, "right": {"state": "hidden"}}
        predicted_eyes = {"left": {"x": 35, "y": 35, "confidence": 0.9}}
        result = evaluate_records([frame("a", [bird("b1", [10, 10, 50, 50], eyes)])], [prediction("a", [detection([10, 10, 50, 50], predicted_eyes)])])
        self.assertEqual((result["eye_tp"], result["eye_fp"], result["eye_fn"]), (0, 1, 1))

    def test_rejects_prediction_from_different_rendition(self) -> None:
        pred = prediction("a", [])
        pred["rendition_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "rendition_sha256"):
            evaluate_records([frame("a", [])], [pred])


if __name__ == "__main__":
    unittest.main()
