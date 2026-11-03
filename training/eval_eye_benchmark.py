#!/usr/bin/env python3
"""Benchmark eye-localization models on CUB held-out species and on a labelled field sample.

Models are given as repeated ``--model`` specs, ``name=type,key=value,...``:

    yolo     YOLO pose weights; eyes=<left_idx>:<right_idx>        e.g. A=yolo,weights=models/eye_pose_v0.pt,eyes=1:2
    pipeline the production PoseLocalizer.localize_eyes() path        e.g. A_pipe=pipeline,weights=models/eye_pose_v0.pt
    simcc    top-down SimCC bird-head ONNX (path required, no default); box=gt|yolo_bird
                                                                     e.g. C=simcc,onnx=<path>,box=yolo_bird

Matching never needs left/right labels: per ground-truth bird, visible eyes are paired with predicted eyes by
minimum total distance. Errors are normalised by the ground-truth bird-box diagonal. On CUB, where sides are
known, an identity-aware score is reported too.

    python -m training.eval_eye_benchmark run --cub data/cub_bird15 --cub-root <CUB_200_2011 folder> \
        --subset a_val --model ... --out data/eye_benchmark/cub_a_val
    python -m training.eval_eye_benchmark run --field data/field_eye_v1 --thresholds <cub metrics.json> \
        --model ... --out data/eye_benchmark/field
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

PCK_THRESHOLDS = (0.02, 0.05, 0.1)
HIT_THRESHOLD = 0.1  # an eye "hit" = matched with normalised error <= 0.1
CUB15_EYES = {"left": 6, "right": 10}  # bird_pose_15 / cub_bird15 keypoint order


# ---------------------------------------------------------------- ground truth

def eye_pose_v0_val_species(species_in_order: list[str], seed: int = 42, val_ratio: float = 0.15) -> set[str]:
    """Replay the species-level split behind eye_pose_v0 (training/convert_cub200.py + prepare_dataset.py).

    ``convert_cub200`` seeds the global RNG and ``prepare_from_coco`` shuffles the species folders (its group
    key is the image's parent folder) in first-appearance order, taking the first ``int(n * val_ratio)``.
    """
    keys = list(dict.fromkeys(species_in_order))
    rng = random.Random(seed)
    rng.shuffle(keys)
    return set(keys[: max(1, int(len(keys) * val_ratio))]) if keys else set()


def cub_species_order(cub_root: Path) -> list[str]:
    rows = (Path(cub_root) / "images.txt").read_text(encoding="utf-8").split("\n")
    return [line.split(maxsplit=1)[1].split("/")[0] for line in rows if line.strip()]


def cub_frames(cub_dir: Path, species: set[str]) -> list[dict]:
    """Frames for CUB images of the given species, from the cub_bird15 YOLO labels (train and val folders)."""
    from PIL import Image

    frames = []
    for split in ("train", "val"):
        for img in sorted((cub_dir / "images" / split).glob("*.jpg")):
            sp = img.stem.split("__")[0]
            if sp not in species:
                continue
            label = cub_dir / "labels" / split / (img.stem + ".txt")
            if not label.is_file():
                continue
            v = [float(x) for x in label.read_text(encoding="utf-8").split()]
            with Image.open(img) as im:
                w, h = im.size
            cx, cy, bw, bh = v[1:5]
            bbox = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            eyes = []
            for side, k in CUB15_EYES.items():
                x, y, vis = v[5 + 3 * k: 8 + 3 * k]
                eyes.append({"state": "visible", "x": x * w, "y": y * h, "side": side} if vis > 0
                            else {"state": "hidden", "side": side})
            frames.append({"image_id": img.stem, "image_path": str(img), "width": w, "height": h, "kind": "bird",
                           "species": sp, "size_stratum": None, "body": "cub",
                           "birds": [{"id": "bird-1", "bbox": bbox, "eyes": eyes}]})
    return frames


def field_frames(sample: Path, birds_only: bool = False) -> list[dict]:
    """Labelled field frames: manifest rows joined with labels.jsonl; excluded and unlabelled frames dropped.

    With ``birds_only``, <sample>/subjects.jsonl ({image_id, bird_index, subject: bird|other}) removes labelled
    subjects that are not birds; a frame whose labelled subjects are all non-birds is dropped entirely (it is
    neither a bird frame nor a clean negative). Frames with no labelled subject stay as negatives.
    """
    man = {r["image_id"]: r for r in _jsonl(sample / "manifest.jsonl")}
    other = set()
    if birds_only:
        tags = _jsonl(sample / "subjects.jsonl")
        if not tags:
            raise ValueError(f"birds_only needs {sample / 'subjects.jsonl'}")
        other = {(t["image_id"], t["bird_index"]) for t in tags if t["subject"] != "bird"}
    frames = []
    for lab in _jsonl(sample / "labels.jsonl"):
        m = man.get(lab["image_id"])
        if m is None or lab.get("excluded"):
            continue
        if lab["rendition_sha256"] != m["rendition_sha256"]:
            raise ValueError(f"{lab['image_id']}: label hash does not match the rendition")
        birds = [b for k, b in enumerate(lab["birds"]) if (lab["image_id"], k) not in other]
        if lab["birds"] and not birds:
            continue
        frames.append({"image_id": m["image_id"], "image_path": m["image_path"], "width": m["width"],
                       "height": m["height"], "kind": "bird" if birds else "negative",
                       "sampled_as": m["kind"], "size_stratum": m.get("size_stratum"), "body": m.get("body"),
                       "year": m.get("year"), "birds": birds})
    return frames


def _jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- matching and metrics

def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_birds(gt_boxes: list, pred_boxes: list, iou_thr: float = 0.3) -> dict[int, int]:
    """Greedy one-to-one gt->pred matching by IoU (pred boxes assumed sorted by confidence)."""
    # highest IoU first; ties go to the earlier (higher-confidence) prediction
    pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gt_boxes) for pi, p in enumerate(pred_boxes)),
                   key=lambda t: (-t[0], t[2], t[1]))
    out, used = {}, set()
    for v, gi, pi in pairs:
        if v < iou_thr or gi in out or pi in used:
            continue
        out[gi] = pi
        used.add(pi)
    return out


def match_eyes(gt_pts: list[tuple[float, float]], pred_pts: list[tuple[float, float]]):
    """Minimum-total-distance pairing of gt and predicted eyes. Returns [(gt_i, pred_j, distance)]."""
    if not gt_pts or not pred_pts:
        return []
    best, best_cost = [], math.inf
    n = min(len(gt_pts), len(pred_pts))
    for gsel in itertools.combinations(range(len(gt_pts)), n):
        for perm in itertools.permutations(range(len(pred_pts)), n):
            pairs = [(gi, pj, math.dist(gt_pts[gi], pred_pts[pj])) for gi, pj in zip(gsel, perm)]
            cost = sum(d for *_, d in pairs)
            if cost < best_cost:
                best, best_cost = pairs, cost
    return best


def bbox_diag(b) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


def frame_records(frame: dict, pred: dict, eye_conf: float, bird_conf: float, iou_thr: float = 0.3) -> dict:
    """Per-frame outcome records consumed by aggregate()."""
    dets = [d for d in pred.get("detections", []) if d["confidence"] >= bird_conf]
    gt = frame["birds"]
    rec = {"image_id": frame["image_id"], "kind": frame["kind"], "size_stratum": frame.get("size_stratum"),
           "body": frame.get("body"), "birds_gt": len(gt), "birds_found": 0, "false_birds": 0,
           "eyes": [], "hidden_scores": [], "false_eyes_on_hidden": 0, "hidden_slots": 0,
           "negative_false_eyes": 0, "identity": [], "closed": []}
    if frame["kind"] == "negative":
        rec["false_birds"] = len(dets)
        rec["negative_false_eyes"] = sum(1 for d in dets for e in d.get("eyes", []) if e["confidence"] >= eye_conf)
        return rec
    m = match_birds([b["bbox"] for b in gt], [d["bbox"] for d in dets], iou_thr)
    rec["birds_found"] = len(m)
    rec["false_birds"] = len(dets) - len(m)
    for gi, bird in enumerate(gt):
        diag = bbox_diag(bird["bbox"])
        vis = [e for e in bird["eyes"] if e["state"] == "visible"]
        n_hidden = sum(1 for e in bird["eyes"] if e["state"] == "hidden")
        rec["hidden_slots"] += n_hidden
        det = dets[m[gi]] if gi in m else None
        all_eyes = sorted(det.get("eyes", []) if det else [], key=lambda e: -e["confidence"])
        kept = [e for e in all_eyes if e["confidence"] >= eye_conf]
        pairs = match_eyes([(e["x"], e["y"]) for e in vis], [(e["x"], e["y"]) for e in kept])
        matched_gt = {gi_ for gi_, _, _ in pairs}
        for gi_, pj, dist in pairs:
            rec["eyes"].append({"error": dist / diag, "confidence": kept[pj]["confidence"], "closed": bool(vis[gi_].get("closed"))})
            if "side" in vis[gi_]:
                rec["identity"].append(kept[pj].get("side") == vis[gi_]["side"] if kept[pj].get("side") else None)
        for gi_ in range(len(vis)):
            if gi_ not in matched_gt:
                rec["eyes"].append({"error": None, "confidence": None, "closed": bool(vis[gi_].get("closed"))})
        extra = len(kept) - len(pairs)
        rec["false_eyes_on_hidden"] += min(extra, n_hidden)
        # Visible-vs-hidden separability: score each slot with the confidence the model put there.
        all_pairs = match_eyes([(e["x"], e["y"]) for e in vis], [(e["x"], e["y"]) for e in all_eyes])
        used = {pj for _, pj, _ in all_pairs}
        leftovers = [e["confidence"] for j, e in enumerate(all_eyes) if j not in used]
        for gi_, pj, dist in all_pairs:
            rec["hidden_scores"].append((1, all_eyes[pj]["confidence"] if dist / diag <= HIT_THRESHOLD else 0.0))
        for _ in range(len(vis) - len(all_pairs)):
            rec["hidden_scores"].append((1, 0.0))
        for k in range(n_hidden):
            rec["hidden_scores"].append((0, leftovers[k] if k < len(leftovers) else 0.0))
        if det and det.get("eye_state") is not None and vis:
            rec["closed"].append({"gt_closed": any(e.get("closed") for e in vis), "pred": det["eye_state"]})
    return rec


def auroc(pairs: list[tuple[int, float]]) -> float | None:
    """Mann-Whitney AUROC via average ranks (ties count half); O(n log n)."""
    n_pos = sum(1 for y, _ in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    ordered = sorted(pairs, key=lambda p: p[1])
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if ordered[k][0] == 1)
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def aggregate(recs: list[dict]) -> dict:
    eyes = [e for r in recs for e in r["eyes"]]
    found = [e["error"] for e in eyes if e["error"] is not None]
    bird_frames = [r for r in recs if r["kind"] == "bird"]
    neg_frames = [r for r in recs if r["kind"] == "negative"]
    frames_with_eye = [r for r in bird_frames if r["eyes"]]
    out = {
        "frames": len(recs), "bird_frames": len(bird_frames), "negative_frames": len(neg_frames),
        "visible_eyes": len(eyes),
        "bird_recall": _ratio(sum(r["birds_found"] for r in bird_frames), sum(r["birds_gt"] for r in bird_frames)),
        "false_birds_per_bird_frame": _ratio(sum(r["false_birds"] for r in bird_frames), len(bird_frames)),
        **{f"pck@{t}": _ratio(sum(1 for e in found if e <= t), len(eyes)) for t in PCK_THRESHOLDS},
        "eye_detected_rate": _ratio(len(found), len(eyes)),
        "median_error": round(sorted(found)[len(found) // 2], 4) if found else None,
        "frame_eye_hit_rate": _ratio(sum(1 for r in frames_with_eye if any(
            e["error"] is not None and e["error"] <= HIT_THRESHOLD for e in r["eyes"])), len(frames_with_eye)),
        "false_eyes_on_hidden_rate": _ratio(sum(r["false_eyes_on_hidden"] for r in recs),
                                            sum(r["hidden_slots"] for r in recs)),
        "visible_vs_hidden_auroc": _round(auroc([p for r in recs for p in r["hidden_scores"]])),
        "negative_frames_with_false_eye": _ratio(sum(1 for r in neg_frames if r["negative_false_eyes"]), len(neg_frames)),
        "negative_false_birds_per_frame": _ratio(sum(r["false_birds"] for r in neg_frames), len(neg_frames)),
    }
    ident = [v for r in recs for v in r["identity"]]
    if ident:
        out["side_correct_rate"] = _ratio(sum(1 for v in ident if v), len(ident))
        out["side_unassigned_rate"] = _ratio(sum(1 for v in ident if v is None), len(ident))
    closed = [c for r in recs for c in r["closed"]]
    if closed:
        out["eye_state_vs_closed_tag"] = {
            "n": len(closed),
            "gt_closed": sum(1 for c in closed if c["gt_closed"]),
            "pred_not_open_given_closed": sum(1 for c in closed if c["gt_closed"] and c["pred"] != "open"),
            "pred_not_open_given_open": sum(1 for c in closed if not c["gt_closed"] and c["pred"] != "open"),
        }
    return out


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def _round(v):
    return round(v, 4) if v is not None else None


BOOT_KEYS = ("bird_recall", "pck@0.05", "pck@0.1", "frame_eye_hit_rate", "false_eyes_on_hidden_rate",
             "visible_vs_hidden_auroc", "negative_frames_with_false_eye")


def bootstrap(recs: list[dict], n: int = 1000, seed: int = 0) -> dict:
    """95% percentile intervals over frames for the headline rates."""
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(n):
        agg = aggregate([recs[rng.randrange(len(recs))] for _ in recs])
        for k in BOOT_KEYS:
            if agg.get(k) is not None:
                samples[k].append(agg[k])
    out = {}
    for k, vals in samples.items():
        vals.sort()
        out[k] = [round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(0.975 * (len(vals) - 1))], 4)]
    return out


def choose_threshold(frames: list[dict], preds: dict[str, dict], bird_conf: float) -> float:
    """Eye-confidence threshold maximising Youden's J (visible vs hidden) on a calibration set."""
    pairs = [p for f in frames for p in frame_records(f, preds.get(f["image_id"], {}), 0.0, bird_conf)["hidden_scores"]]
    pos = sorted(s for y, s in pairs if y == 1)
    neg = sorted(s for y, s in pairs if y == 0)
    if not pos or not neg:
        return 0.5
    best_t, best_j = 0.5, -1.0
    for t in sorted({round(s, 3) for _, s in pairs if s > 0}):
        tpr = sum(1 for s in pos if s >= t) / len(pos)
        fpr = sum(1 for s in neg if s >= t) / len(neg)
        if tpr - fpr > best_j:
            best_t, best_j = t, tpr - fpr
    return best_t


# ---------------------------------------------------------------- model adapters

def parse_model_spec(spec: str) -> dict:
    name, _, rest = spec.partition("=")
    parts = rest.split(",")
    out = {"name": name.strip(), "type": parts[0].strip()}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        out[k.strip()] = v.strip()
    if out["type"] not in ("yolo", "pipeline", "simcc", "topdown"):
        raise ValueError(f"unknown model type in {spec!r}")
    if out["type"] == "simcc" and not out.get("onnx"):
        raise ValueError("simcc models need onnx=<path>")
    return out


class YoloAdapter:
    def __init__(self, spec: dict, device):
        from ultralytics import YOLO

        self.model = YOLO(spec["weights"])
        self.device = device
        left, right = (int(v) for v in spec.get("eyes", "1:2").split(":"))
        self.eyes = {"left": left, "right": right}
        self.imgsz = int(spec.get("imgsz", 640))

    def __call__(self, im, rgb, frame) -> dict:
        r = self.model.predict(im, conf=0.05, max_det=20, imgsz=self.imgsz, verbose=False, device=self.device)[0]
        dets = []
        if r.boxes is not None and r.keypoints is not None:
            xy = r.keypoints.xy.cpu().numpy()
            kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            for i, (b, c) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())):
                eyes = [{"x": float(xy[i][k][0]), "y": float(xy[i][k][1]),
                         "confidence": float(kc[i][k]) if kc is not None else 1.0, "side": side}
                        for side, k in self.eyes.items()]
                dets.append({"bbox": [float(v) for v in b], "confidence": float(c), "eyes": eyes})
        return {"detections": sorted(dets, key=lambda d: -d["confidence"])}


class PipelineAdapter:
    """What scoring consumes: PoseLocalizer.localize_eyes() on the best detection, with its own gates."""

    def __init__(self, spec: dict, device):
        from eye_quality.localization.pose_model import PoseLocalizer

        self.loc = PoseLocalizer(weights=spec["weights"], device=str(device) if device is not None else None)

    def __call__(self, im, rgb, frame) -> dict:
        from eye_quality.schemas import EyeVisibility

        dets = self.loc.predict(frame["image_path"])
        if not dets:
            return {"detections": []}
        locs, _, _ = self.loc.localize_eyes(frame["image_path"], frame["width"], frame["height"])
        eyes = [{"x": e.center_x, "y": e.center_y, "confidence": float(e.confidence), "side": str(e.eye_side)}
                for e in locs if e.visibility not in (EyeVisibility.HIDDEN, EyeVisibility.TOO_SMALL)]
        return {"detections": [{"bbox": list(dets[0].bbox_xyxy), "confidence": dets[0].bbox_confidence,
                                "eyes": eyes}]}


class SimccAdapter:
    EYE_STATES = ("open", "partial", "closed_or_occluded")

    def __init__(self, spec: dict, device):
        from training.teacher.rtmpose_bird_eye import BirdEyeOnnxSession

        self.sess = BirdEyeOnnxSession(spec["onnx"])
        self.box = spec.get("box", "yolo_bird")
        self.pad = float(spec.get("pad", 0.125))
        # stretch (default): padded box resized to 256x256, as in training/teacher/pseudo_label_eye.py; it scored
        # better on CUB than square. square: centred square side = (1 + 2*pad) * longer box side, zero fill
        # outside the frame (the usual top-down RTMPose convention).
        self.crop = spec.get("crop", "stretch")
        if self.crop not in ("square", "stretch"):
            raise ValueError("crop must be square or stretch")
        self.kpt_gate = float(spec.get("kpt_gate", 0.3))
        self.det = None
        if self.box == "yolo_bird":
            from ultralytics import YOLO

            self.det = YOLO(spec.get("det_weights", "models/bird_detect_v1.pt"))
            self.device = device

    def _boxes(self, im, frame):
        if self.box == "gt":
            return [(b["bbox"], 1.0) for b in frame["birds"]]
        r = self.det.predict(im, conf=0.05, max_det=20, verbose=False, device=self.device)[0]
        if r.boxes is None:
            return []
        return [([float(v) for v in b], float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]

    def __call__(self, im, rgb, frame) -> dict:
        import numpy as np

        from training.teacher.eye_identity import crop_resize_rgb, pad_crop_pixels, suggest_teacher_mapping
        from PIL import Image

        from training.teacher.rtmpose_bird_eye import POSE, decode_simcc_outputs, preprocess_crop

        h, w = rgb.shape[:2]
        dets = []
        for box, conf in self._boxes(im, frame):
            if self.crop == "square":
                side = (1 + 2 * self.pad) * max(box[2] - box[0], box[3] - box[1])
                if side <= 1:
                    continue
                x0 = round((box[0] + box[2]) / 2 - side / 2)
                y0 = round((box[1] + box[3]) / 2 - side / 2)
                x1, y1 = x0 + round(side), y0 + round(side)
                # PIL fills the part of the crop outside the frame with zeros
                crop = np.asarray(im.crop((x0, y0, x1, y1)).resize((POSE, POSE), Image.BILINEAR))
                ox, oy, cw, ch = x0, y0, x1 - x0, y1 - y0
            else:
                ox, oy, cw, ch = pad_crop_pixels(np.asarray(box, dtype=np.float64), w, h, self.pad)
                if cw <= 1 or ch <= 1:
                    continue
                crop = crop_resize_rgb(rgb, ox, oy, cw, ch, POSE)
            s = self.sess._session
            raw = dict(zip(self.sess._output_names, s.run(None, {self.sess._input_name: preprocess_crop(crop)})))
            dec = decode_simcc_outputs(raw)
            mapping = suggest_teacher_mapping(dec, kpt_gate=self.kpt_gate)
            eyes = []
            for name in ("eye0", "eye1"):
                px, py, pc = dec.points[name]
                side = mapping.get(name)
                eyes.append({"x": ox + px * cw / POSE, "y": oy + py * ch / POSE, "confidence": float(pc),
                             "side": side.replace("_eye", "") if side else None})
            state = None
            if "eye_state" in raw:
                state = self.EYE_STATES[int(np.argmax(np.asarray(raw["eye_state"]).reshape(-1)[:3]))]
            dets.append({"bbox": list(box), "confidence": conf, "eyes": eyes, "eye_state": state,
                         "orientation": int(dec.orientation_argmax) if dec.orientation_argmax is not None else None})
        return {"detections": sorted(dets, key=lambda d: -d["confidence"])}


class TopDownYoloAdapter:
    """Two-stage: bird boxes from a detector, then a YOLO pose model on each padded box crop."""

    def __init__(self, spec: dict, device):
        from ultralytics import YOLO

        self.det = YOLO(spec.get("det_weights", "models/bird_detect_v1.pt"))
        self.pose = YOLO(spec["weights"])
        self.device = device
        left, right = (int(v) for v in spec.get("eyes", "1:2").split(":"))
        self.eyes = {"left": left, "right": right}
        self.pad = float(spec.get("pad", 0.25))
        self.imgsz = int(spec.get("imgsz", 640))

    def __call__(self, im, rgb, frame) -> dict:
        r = self.det.predict(im, conf=0.05, max_det=20, verbose=False, device=self.device)[0]
        dets = []
        if r.boxes is None:
            return {"detections": dets}
        for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (float(v) for v in box)
            px, py = self.pad * (x2 - x1), self.pad * (y2 - y1)
            cx1, cy1 = max(0, round(x1 - px)), max(0, round(y1 - py))
            cx2, cy2 = min(im.width, round(x2 + px)), min(im.height, round(y2 + py))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            p = self.pose.predict(im.crop((cx1, cy1, cx2, cy2)), conf=0.05, max_det=1, imgsz=self.imgsz,
                                  verbose=False, device=self.device)[0]
            eyes = []
            if p.keypoints is not None and len(p.keypoints.xy):
                xy = p.keypoints.xy[0].cpu().numpy()
                kc = p.keypoints.conf[0].cpu().numpy() if p.keypoints.conf is not None else None
                eyes = [{"x": cx1 + float(xy[k][0]), "y": cy1 + float(xy[k][1]),
                         "confidence": float(kc[k]) if kc is not None else 1.0, "side": side}
                        for side, k in self.eyes.items()]
            dets.append({"bbox": [x1, y1, x2, y2], "confidence": float(conf), "eyes": eyes})
        return {"detections": sorted(dets, key=lambda d: -d["confidence"])}


ADAPTERS = {"yolo": YoloAdapter, "pipeline": PipelineAdapter, "simcc": SimccAdapter, "topdown": TopDownYoloAdapter}


def run_models(frames: list[dict], specs: list[dict], out: Path, device) -> dict[str, dict]:
    """Run every model over every frame; writes <out>/<name>.predictions.jsonl. Returns timing per model."""
    from PIL import Image, ImageOps
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    adapters = {s["name"]: ADAPTERS[s["type"]](s, device) for s in specs}
    timing = {n: [] for n in adapters}
    files = {n: (out / f"{n}.predictions.jsonl").open("w", encoding="utf-8") for n in adapters}
    try:
        for i, f in enumerate(frames, 1):
            with Image.open(f["image_path"]) as src:
                im = ImageOps.exif_transpose(src).convert("RGB")
            rgb = np.asarray(im)
            for name, fn in adapters.items():
                t0 = time.perf_counter()
                p = fn(im, rgb, f)
                timing[name].append(time.perf_counter() - t0)
                files[name].write(json.dumps({"image_id": f["image_id"], **p}) + "\n")
            if i % 200 == 0:
                print(f"{i}/{len(frames)} frames")
    finally:
        for fh in files.values():
            fh.close()
    return {n: {"ms_per_image_median": round(1000 * sorted(t[1:] or t)[len(t[1:] or t) // 2], 1)} for n, t in timing.items()}


def score(frames, specs, out: Path, thresholds: dict[str, float] | None, bird_conf: float, by: list[str]):
    result = {}
    for s in specs:
        preds = {p["image_id"]: p for p in _jsonl(out / f"{s['name']}.predictions.jsonl")}
        thr = (thresholds or {}).get(s["name"])
        calibrated = thr is None
        if calibrated:
            thr = choose_threshold(frames, preds, bird_conf)
        recs = [frame_records(f, preds.get(f["image_id"], {}), thr, bird_conf) for f in frames]
        # Localisation alone: every predicted eye counts, whatever its confidence (false-eye fields meaningless).
        ungated = aggregate([frame_records(f, preds.get(f["image_id"], {}), 0.0, bird_conf) for f in frames])
        entry = {"eye_conf": thr, "eye_conf_chosen_here": calibrated, "overall": aggregate(recs),
                 "ungated": {k: ungated[k] for k in ("pck@0.02", "pck@0.05", "pck@0.1", "median_error",
                                                     "frame_eye_hit_rate")},
                 "ci95": bootstrap(recs)}
        for key in by:
            groups = defaultdict(list)
            for r in recs:
                if r.get(key) is not None:
                    groups[r[key]].append(r)
            entry[f"by_{key}"] = {g: aggregate(v) for g, v in sorted(groups.items())}
        result[s["name"]] = entry
    return result


def markdown(metrics: dict) -> str:
    cols = ["bird_recall", "pck@0.02", "pck@0.05", "pck@0.1", "median_error", "frame_eye_hit_rate",
            "false_eyes_on_hidden_rate", "visible_vs_hidden_auroc", "negative_frames_with_false_eye", "side_correct_rate"]
    lines = ["| Model | eye conf | " + " | ".join(cols) + " | ms/img |", "|---" * (len(cols) + 3) + "|"]
    for name, m in metrics["models"].items():
        o = m["overall"]
        lines.append(f"| {name} | {m['eye_conf']} | " + " | ".join(str(o.get(c)) for c in cols)
                     + f" | {metrics['timing'].get(name, {}).get('ms_per_image_median')} |")
    ucols = ["pck@0.02", "pck@0.05", "pck@0.1", "median_error", "frame_eye_hit_rate"]
    lines += ["", "Ungated (localisation only, every predicted eye counts):", "",
              "| Model | " + " | ".join(ucols) + " |", "|---" * (len(ucols) + 1) + "|"]
    for name, m in metrics["models"].items():
        lines.append(f"| {name} | " + " | ".join(str(m["ungated"].get(c)) for c in ucols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--cub", help="cub_bird15 dataset folder")
    src.add_argument("--field", help="Labelled field sample folder (manifest.jsonl + labels.jsonl)")
    r.add_argument("--cub-root", help="CUB_200_2011 folder (species order for the eye_pose_v0 split replay)")
    r.add_argument("--subset", choices=("b15_val", "a_val", "ab"), default="b15_val",
                   help="b15_val: bird_pose_15 held-out species; a_val: eye_pose_v0 held-out (needs --cub-root); "
                        "ab: held out by both (empty for the current v0 models)")
    r.add_argument("--birds-only", action="store_true",
                   help="Field: drop labelled subjects tagged non-bird in <field>/subjects.jsonl")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--model", action="append", required=True)
    r.add_argument("--thresholds", help="metrics.json from a calibration run; reuse its eye_conf per model")
    r.add_argument("--bird-conf", type=float, default=0.25)
    r.add_argument("--device", default="0")
    r.add_argument("--out", required=True)
    r.add_argument("--score-only", action="store_true", help="Reuse existing predictions in --out")
    a = ap.parse_args()

    specs = [parse_model_spec(s) for s in a.model]
    out = Path(a.out)
    if a.cub:
        cub = Path(a.cub)
        b15_val = set(json.loads((cub / "manifest.json").read_text(encoding="utf-8"))["val_species"])
        species = None
        if a.subset != "b15_val":
            if not a.cub_root:
                raise SystemExit("--cub-root is needed for a_val / ab")
            a_val = eye_pose_v0_val_species(cub_species_order(Path(a.cub_root)))
            species = a_val if a.subset == "a_val" else a_val & b15_val
        frames = cub_frames(cub, species if species is not None else b15_val)
        by = []
        source = {"cub": str(cub), "subset": a.subset, "species": sorted(species if species is not None else b15_val)}
    else:
        frames = field_frames(Path(a.field), birds_only=a.birds_only)
        by = ["size_stratum", "body"]
        source = {"field": a.field, "birds_only": a.birds_only}
    if a.limit:
        frames = frames[: a.limit]
    if not frames:
        raise SystemExit("no frames selected (for CUB, the eye_pose_v0 and bird_pose_15 held-out species may not "
                         "overlap; use --subset a_val or b15_val)")
    print(f"{len(frames)} frames")
    device = None if a.device in ("", "cpu") else (int(a.device) if a.device.isdigit() else a.device)
    timing_path = out / "timing.json"
    if a.score_only:
        timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {}
    else:
        timing = run_models(frames, specs, out, device)
        timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    thresholds = None
    if a.thresholds:
        prior = json.loads(Path(a.thresholds).read_text(encoding="utf-8"))
        thresholds = {n: m["eye_conf"] for n, m in prior["models"].items()}
    metrics = {"source": source, "frames": len(frames), "bird_conf": a.bird_conf, "timing": timing,
               "models": score(frames, specs, out, thresholds, a.bird_conf, by)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(markdown(metrics))


if __name__ == "__main__":
    main()
