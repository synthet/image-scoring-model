#!/usr/bin/env python3
"""Per-part keypoint accuracy of a bird_pose_15 model on the held-out CUB species (data/cub_bird15/val).

For every visible ground-truth part: normalised error = distance(predicted, truth) / ground-truth box diagonal.
PCK@t counts a part as correct when its error is <= t and the model's keypoint confidence is >= --kpt-conf.
A missed bird (no detection) counts every visible part as wrong. Writes JSON and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from training.convert_cub200_bird15 import BIRD15_NAMES


def _labels(path: Path):
    vals = [float(v) for v in path.read_text(encoding="utf-8").split()]
    box = vals[1:5]
    kp = [(vals[5 + 3 * i], vals[6 + 3 * i], vals[7 + 3 * i]) for i in range(len(BIRD15_NAMES))]
    return box, kp


def part_errors(pred_xy, pred_conf, box, kp, width, height, kpt_conf):
    """{part index: normalised error or None (low confidence / missed)} for visible ground-truth parts."""
    diag = math.hypot(box[2] * width, box[3] * height)
    out = {}
    for i, (gx, gy, gv) in enumerate(kp):
        if gv <= 0:
            continue
        if pred_xy is None or pred_conf[i] < kpt_conf:
            out[i] = None
            continue
        out[i] = math.hypot(pred_xy[i][0] - gx * width, pred_xy[i][1] - gy * height) / diag
    return out


def summarise(errors_by_part, thresholds=(0.05, 0.1, 0.2)):
    rows = {}
    for i, name in enumerate(BIRD15_NAMES):
        errs = errors_by_part.get(i, [])
        n = len(errs)
        found = [e for e in errs if e is not None]
        rows[name] = {"visible": n, "detected": len(found),
                      **{f"pck@{t}": round(sum(1 for e in found if e <= t) / n, 4) if n else None for t in thresholds},
                      "median_error": round(sorted(found)[len(found) // 2], 4) if found else None}
    return rows


def evaluate(weights, data, kpt_conf=0.5, limit=None, device=0):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images = sorted((Path(data) / "images" / "val").glob("*.jpg"))[:limit]
    errors = {i: [] for i in range(len(BIRD15_NAMES))}
    missed = 0
    for img in images:
        label = Path(data) / "labels" / "val" / (img.stem + ".txt")
        if not label.is_file():
            continue
        box, kp = _labels(label)
        r = model.predict(str(img), conf=0.25, max_det=1, verbose=False, device=device)[0]
        h, w = r.orig_shape
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            missed += 1
            pred_xy, pred_conf = None, None
        else:
            pred_xy = r.keypoints.xy[0].cpu().numpy().tolist()
            conf = r.keypoints.conf
            pred_conf = conf[0].cpu().numpy().tolist() if conf is not None else [1.0] * len(BIRD15_NAMES)
        for i, e in part_errors(pred_xy, pred_conf, box, kp, w, h, kpt_conf).items():
            errors[i].append(e)
    return {"weights": str(weights), "images": len(images), "missed_birds": missed, "kpt_conf": kpt_conf,
            "parts": summarise(errors)}


def table(result):
    lines = [f"{result['images']} val images, {result['missed_birds']} birds missed, keypoint conf >= "
             f"{result['kpt_conf']}", "", "| Part | Visible | PCK@0.05 | PCK@0.1 | PCK@0.2 | Median error |",
             "|---|---|---|---|---|---|"]
    for name, r in result["parts"].items():
        lines.append(f"| {name} | {r['visible']} | {r['pck@0.05']} | {r['pck@0.1']} | {r['pck@0.2']} | "
                     f"{r['median_error']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/cub_bird15")
    ap.add_argument("--kpt-conf", type=float, default=0.5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", help="write the JSON result here")
    a = ap.parse_args()
    result = evaluate(a.weights, a.data, a.kpt_conf, a.limit)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(table(result))


if __name__ == "__main__":
    main()
