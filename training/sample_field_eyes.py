#!/usr/bin/env python3
"""Sample a stratified, read-only set of library bird frames (plus verified no-bird frames) for eye labelling.

Source photos are only read. Each sampled frame is decoded to a display-oriented RGB rendition (long edge
--long-edge px), written as JPEG under --out/renditions, and listed in --out/manifest.jsonl with its SHA-256,
the source file's size and mtime (to prove it was not modified) and the detector boxes used to pre-fill
the labelling page.

Bird presence is the union of two detectors so the sample does not favour either: the repo's YOLO bird
detector and the open RTMDet-tiny teacher (with its tiled pass). A frame is a bird frame when either one
finds a bird; its size stratum comes from the largest merged box. At most one frame per leaf folder.

    python -m training.sample_field_eyes --root <photo library> --out data/field_eye_v1 \
        --bodies "Camera JPEG,D300,D90,DJI,Z6ii,Z8"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".nef", ".nrw", ".arw", ".cr2", ".cr3", ".dng", ".orf", ".rw2"}
SIZE_EDGES = (0.04, 0.15)  # box area fraction: small < 0.04 <= medium < 0.15 <= large
STRATA = ("small", "medium", "large")
YEAR_RE = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)")


def size_stratum(area_frac: float) -> str:
    if area_frac < SIZE_EDGES[0]:
        return "small"
    if area_frac < SIZE_EDGES[1]:
        return "medium"
    return "large"


def year_of(rel_parts: tuple[str, ...]) -> str:
    for part in rel_parts:
        m = YEAR_RE.search(part)
        if m:
            return m.group(1)
    return "unknown"


def leaf_folders(root: Path, bodies: list[str]) -> list[tuple[str, Path, list[str]]]:
    """(body, folder, image file names) for every folder under root/<body> that holds images."""
    out = []
    for body in bodies:
        base = root / body
        if not base.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(base):
            imgs = sorted(f for f in files if Path(f).suffix.lower() in IMAGE_EXTS)
            if imgs:
                out.append((body, Path(dirpath), imgs))
    return out


def merge_boxes(boxes: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """Greedy merge of normalised x1y1x2y2 boxes from several sources; keeps the higher-confidence box."""
    from training.teacher.pseudo_label_rtmdet import iou

    kept: list[dict] = []
    for b in sorted(boxes, key=lambda d: -d["confidence"]):
        arr = np.array(b["box"], dtype=np.float64)
        if kept and iou(arr, np.array([k["box"] for k in kept])).max() >= iou_thr:
            for k in kept:
                if iou(arr, np.array([k["box"]]))[0] >= iou_thr:
                    k["sources"] = sorted(set(k["sources"]) | set(b["sources"]))
                    break
            continue
        kept.append({**b, "sources": list(b["sources"])})
    return kept


def area_frac(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


class Detectors:
    def __init__(self, yolo_weights: str, rtmdet_ckpt: str, yolo_conf: float, rtmdet_thr: float):
        import torch
        from ultralytics import YOLO

        from training.teacher.rtmdet_tiny import load_upstream

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO(yolo_weights)
        self.rtmdet = load_upstream(rtmdet_ckpt).to(self.device)
        self.yolo_conf = yolo_conf
        self.rtmdet_thr = rtmdet_thr

    def boxes(self, im) -> list[dict]:
        from training.teacher.pseudo_label_rtmdet import BIRD, detect

        w, h = im.size
        out = []
        r = self.yolo.predict(im, conf=self.yolo_conf, max_det=20, verbose=False, device=self.device)[0]
        if r.boxes is not None:
            for xyxy, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                out.append({"box": [float(xyxy[0] / w), float(xyxy[1] / h), float(xyxy[2] / w), float(xyxy[3] / h)],
                            "confidence": round(float(c), 4), "sources": ["yolo_bird"]})
        b, s, c, _, _ = detect(self.rtmdet, im, self.device)
        for box, score, cls in zip(b, s, c):
            if int(cls) == BIRD and float(score) >= self.rtmdet_thr:
                out.append({"box": [float(box[0] / w), float(box[1] / h), float(box[2] / w), float(box[3] / h)],
                            "confidence": round(float(score), 4), "sources": ["rtmdet"]})
        for d in out:
            d["box"] = [round(min(1.0, max(0.0, v)), 5) for v in d["box"]]
        return merge_boxes(out)


def render(src: Path, long_edge: int):
    from PIL import Image

    from training.teacher.pseudo_label_rtmdet import decode

    im = decode(str(src))
    sc = long_edge / max(im.size)
    if sc < 1:
        im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
    return im


def save_rendition(im, out_dir: Path, image_id: str) -> tuple[str, str]:
    import io

    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=93)
    data = buf.getvalue()
    path = out_dir / "renditions" / f"{image_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path.as_posix()), hashlib.sha256(data).hexdigest()


def source_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"source_size_bytes": st.st_size, "source_mtime_ns": st.st_mtime_ns}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="Photo library root (read only)")
    ap.add_argument("--out", required=True, help="Output folder (gitignored, e.g. data/field_eye_v1)")
    ap.add_argument("--bodies", required=True, help="Comma-separated top-level folders to sample")
    ap.add_argument("--birds", type=int, default=120, help="Bird frames, split evenly over size strata")
    ap.add_argument("--negatives", type=int, default=30)
    ap.add_argument("--negatives-manifest", default="data/verified_negatives_v1/manifest.jsonl")
    ap.add_argument("--seed", default="field-eye-v1")
    ap.add_argument("--long-edge", type=int, default=2048)
    ap.add_argument("--max-tries", type=int, default=4000)
    ap.add_argument("--yolo-weights", default="models/bird_detect_v1.pt")
    ap.add_argument("--rtmdet-ckpt", default="models/rtmdet_tiny_coco.pth")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--rtmdet-thr", type=float, default=0.50)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    if manifest_path.exists():
        raise SystemExit(f"{manifest_path} exists; choose a new --out")
    rng = random.Random(a.seed)
    bodies = [b.strip() for b in a.bodies.split(",") if b.strip()]

    negatives = []
    neg_path = Path(a.negatives_manifest)
    if a.negatives and neg_path.is_file():
        rows = [json.loads(line) for line in neg_path.open(encoding="utf-8") if line.strip()]
        rows = [r for r in rows if r.get("label") == "human_verified_no_bird"]
        negatives = rng.sample(rows, min(a.negatives, len(rows)))
    neg_sources = {str(Path(r["source_image_path"])).lower() for r in negatives}

    folders = leaf_folders(root, bodies)
    rng.shuffle(folders)
    by_body: dict[str, list] = {}
    for f in folders:
        by_body.setdefault(f[0], []).append(f)
    print(f"{len(folders)} leaf folders: " + ", ".join(f"{b}={len(v)}" for b, v in sorted(by_body.items())))

    det = Detectors(a.yolo_weights, a.rtmdet_ckpt, a.yolo_conf, a.rtmdet_thr)
    quota = {s: a.birds // len(STRATA) + (1 if i < a.birds % len(STRATA) else 0) for i, s in enumerate(STRATA)}
    got: Counter = Counter()
    rows_out = []
    tries = 0
    skipped = []
    t0 = time.time()
    order = sorted(by_body)
    while sum(got.values()) < a.birds and tries < a.max_tries and any(by_body.values()):
        for body in order:
            if not by_body[body] or sum(got.values()) >= a.birds:
                continue
            _, folder, files = by_body[body].pop()
            src = folder / rng.choice(files)
            if str(src).lower() in neg_sources:
                continue
            tries += 1
            try:
                im = render(src, a.long_edge)
            except Exception as exc:  # unreadable source: skip, never modify
                print(f"skip {src}: {exc}")
                skipped.append(str(src))
                continue
            boxes = det.boxes(im)
            if not boxes:
                continue
            stratum = size_stratum(max(area_frac(b["box"]) for b in boxes))
            if got[stratum] >= quota[stratum]:
                continue
            got[stratum] += 1
            image_id = f"b{len(rows_out) + 1:04d}"
            rpath, rsha = save_rendition(im, out, image_id)
            rel = folder.relative_to(root)
            rows_out.append({
                "schema_version": 1, "image_id": image_id, "kind": "bird", "body": body,
                "year": year_of(rel.parts), "size_stratum": stratum, "group_id": rel.as_posix(),
                "source_path": str(src), **source_fingerprint(src),
                "image_path": rpath, "rendition_sha256": rsha, "width": im.width, "height": im.height,
                "detector_boxes": boxes,
            })
            print(f"[{sum(got.values())}/{a.birds}] {image_id} {body} {stratum} boxes={len(boxes)} "
                  f"tries={tries} {time.time() - t0:.0f}s")

    for i, r in enumerate(negatives, 1):
        src = Path(r["source_image_path"])
        try:
            im = render(src, a.long_edge)
        except Exception as exc:
            print(f"skip negative {src}: {exc}")
            continue
        image_id = f"n{i:04d}"
        rpath, rsha = save_rendition(im, out, image_id)
        rows_out.append({
            "schema_version": 1, "image_id": image_id, "kind": "verified_negative",
            "body": src.relative_to(root).parts[0] if src.is_relative_to(root) else "unknown",
            "year": year_of(src.parts), "size_stratum": None, "group_id": r.get("group_id"),
            "source_path": str(src), **source_fingerprint(src),
            "image_path": rpath, "rendition_sha256": rsha, "width": im.width, "height": im.height,
            "detector_boxes": det.boxes(im),
        })

    with manifest_path.open("w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")
    summary = {
        "seed": a.seed, "tries": tries, "unreadable_skipped": skipped, "bird_frames": sum(got.values()), "by_stratum": dict(got),
        "by_body": dict(Counter(r["body"] for r in rows_out if r["kind"] == "bird")),
        "negatives": sum(1 for r in rows_out if r["kind"] == "verified_negative"),
        "long_edge": a.long_edge, "size_edges": SIZE_EDGES,
    }
    (out / "manifest.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
