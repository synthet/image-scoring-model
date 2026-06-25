#!/usr/bin/env python3
"""Fine-tune Ultralytics YOLO pose for wildlife eye keypoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train wildlife bird pose model")
    parser.add_argument(
        "--data",
        default="training/configs/wildlife_bird.yaml",
        help="Ultralytics dataset YAML",
    )
    parser.add_argument(
        "--base",
        default="yolo11n-pose.pt",
        help="Base pose weights (COCO pretrained)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        default="models/eye_pose_v0.pt",
        help="Copy best checkpoint to this path after training",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a checkpoint (path to last.pt)",
    )
    parser.add_argument(
        "--promote-only",
        default=None,
        help="Skip training: copy this checkpoint (e.g. best.pt) to --output and exit. "
        "Use to publish weights from an interrupted run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2 if sys.platform == "win32" else 8,
        help="Dataloader workers (use 0 if Windows runs out of paging file)",
    )
    parser.add_argument(
        "--cache",
        choices=("none", "ram", "disk"),
        default="disk",
        help="Cache decoded images: disk is safest on Windows; ram is fastest if memory allows",
    )
    args = parser.parse_args()

    if args.promote_only:
        import shutil

        src = Path(args.promote_only).resolve()
        if not src.is_file():
            raise SystemExit(f"Checkpoint to promote not found: {src}")
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        print(f"Promoted {src} -> {out}")
        return

    cache_val: bool | str = False
    if args.cache == "ram":
        cache_val = True
    elif args.cache == "disk":
        cache_val = "disk"

    from ultralytics import YOLO

    repo_root = Path(__file__).resolve().parents[1]
    train_kwargs = {
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(repo_root / "runs" / "pose"),
        "name": "wildlife_bird",
        "exist_ok": True,
        "workers": args.workers,
        "cache": cache_val,
    }
    if args.device:
        train_kwargs["device"] = args.device

    if args.resume:
        ckpt = Path(args.resume).resolve()
        if not ckpt.is_file():
            raise SystemExit(f"Resume checkpoint not found: {ckpt}")
        model = YOLO(str(ckpt))
        results = model.train(resume=True, workers=args.workers, cache=cache_val)
    else:
        model = YOLO(args.base)
        results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", None) or (Path(train_kwargs["project"]) / "wildlife_bird"))
    best = save_dir / "weights" / "best.pt"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if best.is_file():
        import shutil

        shutil.copy2(best, out)
        print(f"Saved best weights to {out}")
    else:
        print(f"Training complete; best weights not found at {best}")


if __name__ == "__main__":
    main()
