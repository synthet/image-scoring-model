"""Bird-eye ONNX teacher: pseudo-label YOLO pose (partial keypoints + mask sidecars).

    python -m training.teacher.pseudo_label_eye detect --pool data/pseudo_eye_v0/pool.json \
        --eye-onnx <local-proprietary-onnx> --det-ckpt models/rtmdet_tiny_coco.pth --out data/pseudo_eye_v0
    python -m training.teacher.pseudo_label_eye materialize --out data/pseudo_eye_v0

Stage ``detect`` runs RTMDet (full frame + 2x2 tiles), then bird-eye ONNX per accepted bird box.
Stage ``materialize`` writes bird labels plus ``labels_mask/`` JSON. Teacher silence is
not proof of a bird-free frame, so negative decisions are audit-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from eye_quality.localization.keypoint_schema import KEYPOINT_NAMES
from training.teacher.eye_alignment import CropTransform, align_keypoints
from training.teacher.eye_identity import (
    crop_resize_rgb,
    orientation_confidence,
    pad_crop_pixels,
    suggest_teacher_mapping,
)
from training.teacher.proprietary_teacher_paths import proprietary_pose_manifest_path
from training.teacher.rtmpose_bird_eye import POSE, BirdEyeOnnxSession

KP_GATE = 0.30
CROP_PAD = 0.125
BIRD_THR, UNCERTAIN_THR = 0.50, 0.35
YOLO_DETECTED_STRATA = {"small_det"}


def _box_norm_to_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    b = np.asarray(box, dtype=np.float64).reshape(4)
    return np.array([b[0] * w, b[1] * h, b[2] * w, b[3] * h], dtype=np.float64)


def _yolo_pose_line(box_norm: np.ndarray, keypoints: np.ndarray) -> str:
    b = np.asarray(box_norm, dtype=np.float64).reshape(4)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    bw, bh = b[2] - b[0], b[3] - b[1]
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for i in range(keypoints.shape[0]):
        x, y, v = keypoints[i]
        parts.append(f"{x:.6f} {y:.6f} {int(v)}")
    return " ".join(parts)


def _run_eye_on_box(rgb: np.ndarray, box_norm: np.ndarray, eye_sess: BirdEyeOnnxSession, kpt_gate: float):
    h, w = rgb.shape[:2]
    xyxy = _box_norm_to_xyxy(box_norm, w, h)
    cx, cy, cw, ch = pad_crop_pixels(xyxy, w, h, CROP_PAD)
    crop = crop_resize_rgb(rgb, cx, cy, cw, ch, POSE)
    decoded = eye_sess.run(crop)
    x2, y2 = cx + cw, cy + ch
    transform = CropTransform.from_crop((w, h), (cx, cy, x2, y2), (POSE, POSE), letterbox=False)
    mapping = suggest_teacher_mapping(decoded, kpt_gate=kpt_gate)
    partial = align_keypoints(decoded.points, mapping, transform, min_confidence=kpt_gate)
    orient, oconf = orientation_confidence(decoded.orientation_logits)
    return {
        "mapping": mapping,
        "orientation": orient,
        "orientation_confidence": round(oconf, 4),
        "keypoints_norm": partial.keypoints.tolist(),
        "annotation_known": partial.annotation_known.tolist(),
        "teacher_points_crop": {k: list(v) for k, v in decoded.points.items()},
    }


def cmd_detect(a):
    import torch

    from training.teacher.pseudo_label_rtmdet import LONG_EDGE, SAVE_EDGE, decide, decode, detect as rtmdet_detect
    from training.teacher.rtmdet_tiny import load_upstream

    out = Path(a.out)
    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))["pool"]
    if a.limit:
        pool = pool[: int(a.limit)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = load_upstream(a.det_ckpt).to(device)
    eye_sess = BirdEyeOnnxSession(a.eye_onnx)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    audit_path = out / "teacher_eye_detections.jsonl"
    done = set()
    if audit_path.exists():
        done = {json.loads(line)["image_id"] for line in audit_path.open(encoding="utf-8")}
    audit = audit_path.open("a", encoding="utf-8")
    t0 = time.time()
    for n, r in enumerate(pool, 1):
        if r["image_id"] in done:
            continue
        rec = {k: r[k] for k in ("image_id", "stratum", "folder_id", "birds_kw", "animal_kw")}
        try:
            im = decode(r["path"])
            sc = LONG_EDGE / max(im.size)
            if sc < 1:
                im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
            rgb = np.asarray(im)
            b, s, c, w, h = rtmdet_detect(det, im, device)
            save = im.copy()
            save.thumbnail((SAVE_EDGE, SAVE_EDGE), Image.LANCZOS)
            save.save(out / "cache" / f"{r['image_id']}.jpg", quality=92)
            boxes_n = np.round(b / [w, h, w, h], 5) if len(b) else np.zeros((0, 4))
            decision, keep = decide(
                boxes_n, s, c, r["birds_kw"], r["animal_kw"], a.bird_thr, a.uncertain_thr, a.min_area,
            )
            instances = []
            if decision == "bird":
                for i in keep:
                    inst = _run_eye_on_box(rgb, boxes_n[i], eye_sess, a.kpt_gate)
                    inst["box_norm"] = boxes_n[i].tolist()
                    instances.append(inst)
            rec.update(
                w=w,
                h=h,
                decision=decision,
                det_boxes=boxes_n.tolist(),
                det_scores=np.round(s, 4).tolist(),
                det_classes=c.tolist(),
                instances=instances,
                eye_onnx=str(Path(a.eye_onnx).resolve()),
            )
        except Exception as ex:  # noqa: BLE001
            rec["error"] = str(ex)[:300]
        audit.write(json.dumps(rec) + "\n")
        audit.flush()
        print(f"\r{n}/{len(pool)} {time.time() - t0:.0f}s", end="", file=sys.stderr)
    print(file=sys.stderr)


def cmd_materialize(a):
    import shutil
    from collections import Counter

    out = Path(a.out)
    recs = [json.loads(line) for line in (out / "teacher_eye_detections.jsonl").open(encoding="utf-8")]
    if any(
        any(inst["annotation_known"][1:3])
        for rec in recs for inst in rec.get("instances", [])
    ):
        raise ValueError(
            "Eye labels require reviewed anatomical identity; keep detections for audit."
        )
    for sub in ("images", "labels", "labels_mask"):
        shutil.rmtree(out / sub, ignore_errors=True)
    counts = Counter()
    boxes_n = Counter()
    for r in recs:
        if "error" in r:
            counts[("error", r.get("stratum"))] += 1
            continue
        decision = r["decision"]
        if decision == "negative" and r["stratum"] in YOLO_DETECTED_STRATA:
            decision = "uncertain_student_saw_bird"
        counts[(decision, r["stratum"])] += 1
        if decision != "bird":
            continue
        split = (
            "val"
            if int(hashlib.md5(str(r["folder_id"]).encode()).hexdigest(), 16) % 100 < a.val_frac * 100
            else "train"
        )
        for sub in ("images", "labels", "labels_mask"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
        iid = r["image_id"]
        shutil.copyfile(out / "cache" / f"{iid}.jpg", out / "images" / split / f"{iid}.jpg")
        lines = []
        mask_instances = []
        if decision == "bird":
            for inst in r.get("instances", []):
                kpts = np.array(inst["keypoints_norm"], dtype=np.float64)
                known = np.array(inst["annotation_known"], dtype=bool)
                for j in range(len(KEYPOINT_NAMES)):
                    if not known[j]:
                        kpts[j] = (0.0, 0.0, 0.0)
                lines.append(_yolo_pose_line(inst["box_norm"], kpts))
                mask_instances.append(
                    dict(
                        box_norm=inst["box_norm"],
                        annotation_known=inst["annotation_known"],
                        mapping=inst.get("mapping"),
                        orientation=inst.get("orientation"),
                    )
                )
                boxes_n[split] += 1
        (out / "labels" / split / f"{iid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if mask_instances:
            (out / "labels_mask" / split / f"{iid}.json").write_text(
                json.dumps(dict(frame_decision=decision, instances=mask_instances), indent=0),
                encoding="utf-8",
            )
    pose_manifest = None
    pose_path = Path(a.pose_manifest)
    if pose_path.is_file():
        pose_manifest = json.loads(pose_path.read_text(encoding="utf-8"))
    det_manifest = None
    det_path = Path(a.det_manifest)
    if det_path.is_file():
        det_manifest = json.loads(det_path.read_text(encoding="utf-8"))
    manifest = dict(
        policy=dict(
            bird_thr=a.bird_thr,
            uncertain_thr=a.uncertain_thr,
            min_area=a.min_area,
            kpt_gate=a.kpt_gate,
            crop_pad=CROP_PAD,
            val_frac=a.val_frac,
        ),
        teachers=dict(pose=pose_manifest, detect=det_manifest),
        counts={f"{k[0]}|{k[1]}": v for k, v in sorted(counts.items())},
        pose_instances=dict(boxes_n),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    by_dec = Counter()
    for (d, _), v in counts.items():
        by_dec[d] += v
    print(json.dumps(dict(by_decision=by_dec, pose_instances=boxes_n)))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("detect")
    d.add_argument("--pool", required=True)
    d.add_argument("--eye-onnx", required=True, help="Local proprietary bird-eye ONNX (not committed)")
    d.add_argument("--det-ckpt", default="models/rtmdet_tiny_coco.pth")
    d.add_argument("--out", required=True)
    d.add_argument("--bird-thr", type=float, default=BIRD_THR)
    d.add_argument("--uncertain-thr", type=float, default=UNCERTAIN_THR)
    d.add_argument("--min-area", type=float, default=0.0005)
    d.add_argument("--kpt-gate", type=float, default=KP_GATE)
    d.add_argument("--limit", type=int, default=None, help="Pilot cap: first N pool rows")
    m = sp.add_parser("materialize")
    m.add_argument("--out", required=True)
    m.add_argument("--bird-thr", type=float, default=BIRD_THR)
    m.add_argument("--uncertain-thr", type=float, default=UNCERTAIN_THR)
    m.add_argument("--min-area", type=float, default=0.0005)
    m.add_argument("--kpt-gate", type=float, default=KP_GATE)
    m.add_argument("--val-frac", type=float, default=0.1)
    m.add_argument(
        "--pose-manifest",
        default=None,
        help="Proprietary pose manifest JSON (default: docs/private or .docs/private when present)",
    )
    m.add_argument("--det-manifest", default="models/rtmdet_tiny_coco.manifest.json")
    a = ap.parse_args()
    if a.cmd == "materialize" and a.pose_manifest is None:
        a.pose_manifest = str(proprietary_pose_manifest_path())
    {"detect": cmd_detect, "materialize": cmd_materialize}[a.cmd](a)


if __name__ == "__main__":
    main()
