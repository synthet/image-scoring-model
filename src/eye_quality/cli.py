"""CLI for eye quality scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eye_quality.localization.pose_model import resolve_weights_path
from eye_quality.pipeline import PipelineConfig, score_image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eye-quality", description="Wildlife eye focus scoring")
    sub = parser.add_subparsers(dest="command", required=True)

    score_p = sub.add_parser("score", help="Score a single image")
    score_p.add_argument("path", type=str, help="Path to image file")
    score_p.add_argument("--weights", type=str, default=None)
    score_p.add_argument("--device", type=str, default=None)
    score_p.add_argument("--imgsz", type=int, default=640)
    score_p.add_argument("--debug-dir", type=str, default=None)
    score_p.add_argument("--output", type=str, default=None, help="Write JSON to file")

    batch_p = sub.add_parser("batch", help="Score images listed in a JSONL manifest")
    batch_p.add_argument("manifest", type=str, help="JSONL file with {\"path\": \"...\"} per line")
    batch_p.add_argument("--weights", type=str, default=None)
    batch_p.add_argument("--device", type=str, default=None)
    batch_p.add_argument("--imgsz", type=int, default=640)
    batch_p.add_argument("--debug-dir", type=str, default=None)
    batch_p.add_argument("--output", type=str, required=True, help="Output JSONL path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = PipelineConfig(
        weights=resolve_weights_path(args.weights),
        device=args.device,
        imgsz=args.imgsz,
        debug_dir=args.debug_dir,
    )

    if args.command == "score":
        path = Path(args.path)
        if not path.is_file():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1
        result = score_image(path, config=config)
        payload = json.dumps(result.to_api_dict(), indent=2)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return 0

    if args.command == "batch":
        manifest = Path(args.manifest)
        if not manifest.is_file():
            print(f"Error: manifest not found: {manifest}", file=sys.stderr)
            return 1
        out_path = Path(args.output)
        lines_out: list[str] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img_path = row.get("path") or row.get("image_path")
            if not img_path:
                continue
            if not Path(img_path).is_file():
                record = {"path": img_path, "error": "file_not_found"}
            else:
                result = score_image(img_path, config=config)
                record = {"path": img_path, **result.to_api_dict()}
            lines_out.append(json.dumps(record))
        out_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
