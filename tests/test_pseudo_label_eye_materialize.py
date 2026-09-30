import json
from pathlib import Path

import pytest

from training.teacher.pseudo_label_eye import cmd_materialize


def _pose_manifest_stub(path: Path) -> None:
    path.write_text(
        json.dumps({"licence": "proprietary", "training_use_permission": {"status": "granted"}}),
        encoding="utf-8",
    )


def test_materialize_writes_pose_label_and_mask(tmp_path):
    out = tmp_path / "pseudo"
    pose_manifest = tmp_path / "pose.manifest.json"
    _pose_manifest_stub(pose_manifest)
    cache = out / "cache"
    cache.mkdir(parents=True)
    (cache / "42.jpg").write_bytes(b"jpeg")
    rec = {
        "image_id": 42,
        "folder_id": 7,
        "stratum": "random",
        "decision": "bird",
        "instances": [
            {
                "box_norm": [0.2, 0.2, 0.5, 0.5],
                "keypoints_norm": [[0.3, 0.3, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "annotation_known": [True, False, False, False, False, False],
                "mapping": {"beak": "beak", "eye0": None, "eye1": None},
            }
        ],
    }
    (out / "teacher_eye_detections.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

    from types import SimpleNamespace

    cmd_materialize(
        SimpleNamespace(
            out=str(out),
            bird_thr=0.5,
            uncertain_thr=0.35,
            min_area=0.0005,
            kpt_gate=0.3,
            val_frac=0.0,
            pose_manifest=str(pose_manifest),
            det_manifest="models/rtmdet_tiny_coco.manifest.json",
        )
    )
    label = (out / "labels" / "train" / "42.txt").read_text(encoding="utf-8")
    assert label.startswith("0 ")
    assert " 2 " in label
    mask = json.loads((out / "labels_mask" / "train" / "42.json").read_text(encoding="utf-8"))
    assert mask["instances"][0]["annotation_known"][1] is False


def test_teacher_silent_frame_is_not_materialized_as_verified_negative(tmp_path):
    out = tmp_path / "pseudo"
    pose_manifest = tmp_path / "pose.manifest.json"
    _pose_manifest_stub(pose_manifest)
    cache = out / "cache"
    cache.mkdir(parents=True)
    (cache / "42.jpg").write_bytes(b"jpeg")
    (out / "teacher_eye_detections.jsonl").write_text(
        json.dumps({"image_id": 42, "folder_id": 7, "stratum": "negative", "decision": "negative"}) + "\n",
        encoding="utf-8",
    )
    from types import SimpleNamespace

    cmd_materialize(
        SimpleNamespace(
            out=str(out), bird_thr=0.5, uncertain_thr=0.35, min_area=0.0005,
            kpt_gate=0.3, val_frac=0.0,
            pose_manifest=str(pose_manifest),
            det_manifest="models/rtmdet_tiny_coco.manifest.json",
        )
    )
    assert not (out / "labels" / "train" / "42.txt").exists()
    assert not (out / "images" / "train" / "42.jpg").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["negative|negative"] == 1


def test_unaudited_eye_candidate_is_rejected_before_existing_data_is_replaced(tmp_path):
    out = tmp_path / "pseudo"
    pose_manifest = tmp_path / "pose.manifest.json"
    _pose_manifest_stub(pose_manifest)
    (out / "cache").mkdir(parents=True)
    (out / "cache" / "42.jpg").write_bytes(b"jpeg")
    old_label = out / "labels" / "train" / "keep.txt"
    old_label.parent.mkdir(parents=True)
    old_label.write_text("existing", encoding="utf-8")
    rec = {
        "image_id": 42, "folder_id": 7, "stratum": "random", "decision": "bird",
        "instances": [{
            "box_norm": [0.2, 0.2, 0.5, 0.5],
            "keypoints_norm": [[0.3, 0.3, 2], [0.3, 0.3, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "annotation_known": [True, True, False, False, False, False],
        }],
    }
    (out / "teacher_eye_detections.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    from types import SimpleNamespace

    args = SimpleNamespace(
        out=str(out), bird_thr=0.5, uncertain_thr=0.35, min_area=0.0005,
        kpt_gate=0.3, val_frac=0.0,
        pose_manifest=str(pose_manifest),
        det_manifest="models/rtmdet_tiny_coco.manifest.json",
    )
    with pytest.raises(ValueError, match="Eye labels require"):
        cmd_materialize(args)
    assert old_label.read_text(encoding="utf-8") == "existing"
