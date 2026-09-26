"""RTMDet-tiny (upstream COCO) as a teacher: pseudo-label bird boxes in YOLO format for bird_detect_v0 and the
pose model's detection stage.

    python -m training.teacher.pseudo_label_rtmdet detect --pool data/pseudo_rtmdet_v0/pool.json \
        --ckpt models/rtmdet_tiny_coco.pth --out data/pseudo_rtmdet_v0
    python -m training.teacher.pseudo_label_rtmdet materialize --out data/pseudo_rtmdet_v0

Per frame: decode (display-oriented), resize to a 2048 px long edge, run the teacher on the full frame and on
2x2 overlapping tiles (for small birds), merge, then apply the label policy:

  bird box      COCO `bird` >= BIRD_THR only (keywords never promote another class); boxes >= 70% inside a
                higher-scoring accepted box are dropped as tile/full-frame duplicates
  uncertain     a non-bird animal >= BIRD_THR, or any animal box in [UNCERTAIN_THR, BIRD_THR) not covered by an
                accepted box -> frame excluded (no false-negative training)
  negative      no animal box >= UNCERTAIN_THR and no animal-related keyword -> empty label file; never for
                frames where bird_detect_v0 itself found a bird (YOLO_DETECTED_STRATA)
  otherwise     (keyword says animal, teacher finds none) -> excluded

Two stages: `detect` runs the teacher once and caches boxes; `materialize` applies the policy offline, so
thresholds can be re-tuned without re-inference.

Split: train/val by folder (hash), so no folder leaks across splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from training.teacher.rtmdet_tiny import letterbox, load_upstream, postprocess

BIRD, ANIMALS = 14, set(range(14, 24))
# Calibrated on the owner-labelled #377 cohort (eval only): 0.50 / 0.35 gave 1.9% bird-label FP and
# 0 negatives on bird frames. See data/<set>/manifest.json for the policy actually used.
BIRD_THR, UNCERTAIN_THR = 0.50, 0.35
YOLO_DETECTED_STRATA = {"small_det"}
LONG_EDGE, SAVE_EDGE = 2048, 1280
TILE_FRAC, INNER_EDGE_TOL = 0.6, 0.01


def decode(path: str) -> Image.Image:
    """Display-oriented RGB. Uses the backend's decoder when importable (same route as the benchmarks)."""
    try:
        from modules.thumbnails import bake_orientation, open_rendition_for_ml
        img, _ = open_rendition_for_ml(path)
        return bake_orientation(img.convert("RGB"), path)
    except ImportError:
        from PIL import ImageOps
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def run_teacher(model, rgb: np.ndarray, device):
    x, sc, px, py = letterbox(rgb)
    with torch.no_grad():
        b, s = model(x.to(device))
    bb, ss, cc = postprocess(b.cpu(), s.cpu(), score_thr=0.05)
    bb = (bb - torch.tensor([px, py, px, py])) / sc
    return bb.numpy(), ss.numpy(), cc.numpy()


def detect(model, im: Image.Image, device):
    W, H = im.size
    rgb = np.asarray(im)
    boxes, scores, classes = [], [], []
    b, s, c = run_teacher(model, rgb, device)
    boxes.append(b)
    scores.append(s)
    classes.append(c)
    tw, th = int(W * TILE_FRAC), int(H * TILE_FRAC)
    for ox in (0, W - tw):
        for oy in (0, H - th):
            b, s, c = run_teacher(model, rgb[oy:oy + th, ox:ox + tw], device)
            keep = []
            for i, (x1, y1, x2, y2) in enumerate(b):
                inner = ((ox > 0 and x1 <= INNER_EDGE_TOL * tw) or (ox == 0 and x2 >= tw * (1 - INNER_EDGE_TOL))
                         or (oy > 0 and y1 <= INNER_EDGE_TOL * th) or (oy == 0 and y2 >= th * (1 - INNER_EDGE_TOL)))
                if not inner:
                    keep.append(i)
            boxes.append(b[keep] + np.array([ox, oy, ox, oy]))
            scores.append(s[keep])
            classes.append(c[keep])
    b, s, c = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
    animal = np.isin(c, list(ANIMALS))
    b, s, c = b[animal], s[animal], c[animal]
    if len(b):
        k = nms(torch.from_numpy(b).float(), torch.from_numpy(s).float(), 0.5).numpy()
        b, s, c = b[k], s[k], c[k]
    return b, s, c, W, H


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return i / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i, 1e-9)


def dedupe_contained(b, s, keep, frac=0.7):
    """Drop an accepted box when >= frac of it lies inside a higher-scoring accepted box (tile + full-frame
    duplicates, e.g. a wader boxed once with and once without its reflection)."""
    out = []
    for i in sorted(keep, key=lambda k: -s[k]):
        ai = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        dup = False
        for j in out:
            ix = max(0.0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
            iy = max(0.0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
            aj = (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1])
            if ix * iy >= frac * min(ai, aj):
                dup = True
                break
        if not dup:
            out.append(i)
    return out


def decide(b, s, c, birds_kw, animal_kw, bird_thr=BIRD_THR, unc_thr=UNCERTAIN_THR, min_area=0.0):
    """b: normalized x1y1x2y2. Boxes smaller than min_area (fraction of frame) are ignored entirely."""
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)
    idx = [i for i in range(len(b)) if area[i] >= min_area]
    # Only the COCO `bird` class makes a positive. The `birds` keyword is itself unreliable (CLIP tags), so a
    # confident non-bird animal (rabbit, cat, or a raptor COCO calls `bear`) makes the frame uncertain instead.
    acc = [i for i in idx if s[i] >= bird_thr and c[i] == BIRD]
    if any(s[i] >= bird_thr and c[i] != BIRD for i in idx):
        return "uncertain_mammal", []
    for i in idx:
        if unc_thr <= s[i] < bird_thr and (not acc or iou(b[i], b[acc]).max() < 0.5):
            return "uncertain_low_conf", []
    if acc:
        return "bird", dedupe_contained(b, s, acc)
    if not animal_kw and not any(s[i] >= unc_thr for i in idx):
        return "negative", []
    return "uncertain_keyword_only", []


def cmd_detect(a):
    out = Path(a.out)
    pool = json.loads(Path(a.pool).read_text())["pool"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_upstream(a.ckpt).to(device)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    audit_path = out / "teacher_detections.jsonl"
    done = {json.loads(line)["image_id"] for line in audit_path.open()} if audit_path.exists() else set()
    audit = audit_path.open("a")
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
            b, s, c, W, H = detect(model, im, device)
            save = im.copy()
            save.thumbnail((SAVE_EDGE, SAVE_EDGE), Image.LANCZOS)
            save.save(out / "cache" / f"{r['image_id']}.jpg", quality=92)
            rec.update(w=W, h=H, boxes=np.round(b / [W, H, W, H], 5).tolist() if len(b) else [],
                       scores=np.round(s, 4).tolist(), classes=c.tolist())
        except Exception as ex:  # noqa: BLE001
            rec["error"] = str(ex)[:200]
        audit.write(json.dumps(rec) + "\n")
        audit.flush()
        print(f"\r{n}/{len(pool)} {time.time() - t0:.0f}s", end="", file=sys.stderr)
    print(file=sys.stderr)


def cmd_materialize(a):
    import shutil
    from collections import Counter
    out = Path(a.out)
    recs = [json.loads(line) for line in (out / "teacher_detections.jsonl").open()]
    for sub in ("images", "labels"):
        shutil.rmtree(out / sub, ignore_errors=True)
    counts, boxes_n = Counter(), Counter()
    for r in recs:
        if "error" in r:
            counts[("error", r.get("stratum"))] += 1
            continue
        b = np.array(r["boxes"], dtype=float).reshape(-1, 4)
        s, c = np.array(r["scores"]), np.array(r["classes"])
        decision, keep = decide(b, s, c, r["birds_kw"], r["animal_kw"], a.bird_thr, a.uncertain_thr, a.min_area)
        if decision == "negative" and r["stratum"] in YOLO_DETECTED_STRATA:
            decision = "uncertain_student_saw_bird"  # never teach the student to ignore a bird it found
        counts[(decision, r["stratum"])] += 1
        if decision not in ("bird", "negative"):
            continue
        split = "val" if int(hashlib.md5(str(r["folder_id"]).encode()).hexdigest(), 16) % 100 < a.val_frac * 100 else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "cache" / f"{r['image_id']}.jpg", out / "images" / split / f"{r['image_id']}.jpg")
        lines = [f"0 {(b[i, 0] + b[i, 2]) / 2:.6f} {(b[i, 1] + b[i, 3]) / 2:.6f} {b[i, 2] - b[i, 0]:.6f} {b[i, 3] - b[i, 1]:.6f}"
                 for i in keep]
        boxes_n[split] += len(lines)
        (out / "labels" / split / f"{r['image_id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    manifest = dict(policy=dict(bird_thr=a.bird_thr, uncertain_thr=a.uncertain_thr, min_area=a.min_area,
                                val_frac=a.val_frac, tiles="2x2 @ 60%", long_edge=LONG_EDGE, save_edge=SAVE_EDGE),
                    teacher=json.loads(Path(a.teacher_manifest).read_text()) if a.teacher_manifest else None,
                    counts={f"{k[0]}|{k[1]}": v for k, v in sorted(counts.items())}, boxes=dict(boxes_n))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    by_dec = Counter()
    for (d, _), v in counts.items():
        by_dec[d] += v
    print(json.dumps(dict(by_decision=by_dec, boxes=boxes_n)))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("detect")
    d.add_argument("--pool", required=True)
    d.add_argument("--ckpt", required=True)
    d.add_argument("--out", required=True)
    m = sp.add_parser("materialize")
    m.add_argument("--out", required=True)
    m.add_argument("--bird-thr", type=float, default=BIRD_THR)
    m.add_argument("--uncertain-thr", type=float, default=UNCERTAIN_THR)
    m.add_argument("--min-area", type=float, default=0.0005)
    m.add_argument("--val-frac", type=float, default=0.1)
    m.add_argument("--teacher-manifest", default="models/rtmdet_tiny_coco.manifest.json")
    a = ap.parse_args()
    {"detect": cmd_detect, "materialize": cmd_materialize}[a.cmd](a)


if __name__ == "__main__":
    main()
