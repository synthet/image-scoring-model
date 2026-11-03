#!/usr/bin/env python3
"""Local eye-labelling page for a field sample (see training/sample_field_eyes.py).

Serves index.html, the manifest, the renditions it lists and a label API on 127.0.0.1 only. Labels are kept
in <sample>/labels.jsonl (one row per image, rewritten atomically on every save). Source photos are never
touched: the page only reads the renditions the sampler wrote.

    python -m training.label_eyes.serve data/field_eye_v1 --port 8765
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".labels-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def validate_label(row: dict, manifest: dict[str, dict]) -> str | None:
    m = manifest.get(row.get("image_id"))
    if m is None:
        return "unknown image_id"
    if row.get("rendition_sha256") != m["rendition_sha256"]:
        return "rendition hash mismatch"
    if row.get("excluded"):
        return None
    for bird in row.get("birds", []):
        x1, y1, x2, y2 = bird.get("bbox", [0, 0, 0, 0])
        if not (0 <= x1 < x2 <= m["width"] and 0 <= y1 < y2 <= m["height"]):
            return f"bird {bird.get('id')}: bbox outside the image"
        eyes = bird.get("eyes", [])
        if len(eyes) != 2:
            return f"bird {bird.get('id')}: needs exactly two eye slots"
        for e in eyes:
            if e.get("state") not in ("visible", "hidden", "unresolvable", "unknown"):
                return f"bird {bird.get('id')}: eye state missing"
            if e["state"] == "visible" and not ("x" in e and "y" in e):
                return f"bird {bird.get('id')}: visible eye without a point"
    return None


def make_handler(sample: Path, manifest_rows: list[dict]):
    manifest = {r["image_id"]: r for r in manifest_rows}
    labels_path = sample / "labels.jsonl"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - quiet; signature matches the base class
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/manifest":
                public = [{k: r[k] for k in ("image_id", "kind", "body", "year", "size_stratum", "width", "height",
                                             "rendition_sha256", "detector_boxes")} for r in manifest_rows]
                with LOCK:
                    labels = {r["image_id"]: r for r in load_jsonl(labels_path)}
                self._json(200, {"images": public, "labels": labels})
            elif self.path.startswith("/img/"):
                m = manifest.get(self.path[len("/img/"):])
                if m is None:
                    self._json(404, {"error": "not found"})
                    return
                self._send(200, Path(m["image_path"]).read_bytes(), "image/jpeg")
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/label":
                self._json(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", "0"))
            try:
                row = json.loads(self.rfile.read(n))
            except json.JSONDecodeError:
                self._json(400, {"error": "bad json"})
                return
            err = validate_label(row, manifest)
            if err:
                self._json(400, {"error": err})
                return
            row["schema_version"] = 1
            row["reviewed"] = True
            with LOCK:
                rows = {r["image_id"]: r for r in load_jsonl(labels_path)}
                rows[row["image_id"]] = row
                order = [r["image_id"] for r in manifest_rows if r["image_id"] in rows]
                write_jsonl_atomic(labels_path, [rows[i] for i in order])
            self._json(200, {"ok": True, "labelled": len(order)})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sample", help="Sample folder holding manifest.jsonl")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    sample = Path(a.sample)
    rows = load_jsonl(sample / "manifest.jsonl")
    if not rows:
        raise SystemExit(f"no manifest rows in {sample}")
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(sample, rows))
    print(f"Labelling {len(rows)} images: http://127.0.0.1:{a.port}/  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
