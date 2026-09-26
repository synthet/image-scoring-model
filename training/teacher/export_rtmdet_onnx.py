"""Export the upstream RTMDet-tiny (COCO) checkpoint to ONNX with a provenance manifest.

    python -m training.teacher.export_rtmdet_onnx --ckpt models/rtmdet_tiny_coco.pth --out models/rtmdet_tiny_coco.onnx

Outputs raw per-prior predictions (no NMS in-graph), so any runtime can apply its own post-processing:
  boxes  (1, 8400, 4)  x1,y1,x2,y2 in 640x640 letterbox pixels
  scores (1, 8400, 80) sigmoid class scores
Input: BGR float32 (1, 3, 640, 640), normalized with the upstream mean/std (see rtmdet_tiny.MEAN/STD).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from training.teacher.rtmdet_tiny import MEAN, STD, load_upstream

UPSTREAM_URL = ("https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/"
                "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth")
EXPECTED_SHA256_PREFIX = "78e30dcc"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    ckpt, out = Path(a.ckpt), Path(a.out)
    ck_sha = sha256(ckpt)
    if not ck_sha.startswith(EXPECTED_SHA256_PREFIX):
        raise SystemExit(f"checkpoint sha256 {ck_sha[:12]} does not match the upstream {EXPECTED_SHA256_PREFIX}")
    model = load_upstream(str(ckpt))
    torch.onnx.export(model, torch.zeros(1, 3, 640, 640), str(out), opset_version=a.opset,
                      input_names=["input"], output_names=["boxes", "scores"], dynamo=False)
    manifest = {
        "model": "RTMDet-tiny COCO (80 classes)",
        "licence": "Apache-2.0 (OpenMMLab mmdetection)",
        "upstream_url": UPSTREAM_URL,
        "checkpoint_sha256": ck_sha,
        "onnx_sha256": sha256(out),
        "module": "training/teacher/rtmdet_tiny.py",
        "module_sha256": sha256(Path(__file__).with_name("rtmdet_tiny.py")),
        "opset": a.opset,
        "input": {"layout": "NCHW", "channels": "BGR", "size": 640, "mean": MEAN, "std": STD,
                  "letterbox": "uniform scale, centred, pad 114"},
        "outputs": {"boxes": "(1,8400,4) x1y1x2y2 letterbox px", "scores": "(1,8400,80) sigmoid"},
        "postprocess": {"pre_top_k": 5000, "nms_iou": 0.65, "score_thr": 0.001, "per_class": 200, "keep": 300},
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("checkpoint_sha256", "onnx_sha256")}, indent=2))


if __name__ == "__main__":
    main()
