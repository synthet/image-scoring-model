#!/usr/bin/env python3
"""Build an eye-crop dataset from species-labeled NEFs in the backend's Postgres DB.

Pipeline (MVP-2 data prep):

  1. Query ``image-scoring-backend`` Postgres for images carrying a ``species:*``
     keyword (i.e. bird images already identified by BioCLIP).
  2. Translate each stored WSL path (``/mnt/d/Photos/...``) to a Windows path and
     extract the **full-resolution** embedded JPEG preview from the NEF (Nikon
     stores an ~8256x5504 preview; this is the native resolution we crop from --
     NOT the 512px gallery thumbnail).
  3. Run the validated Stage-1 ``PoseLocalizer`` to find eye keypoints and crop each
     eye at native resolution via ``crop_eye_region``.
  4. Write the crops plus a JSONL manifest mapping every crop back to its source
     image (DB id, species, confidence, eye side, visibility, keypoint coords,
     subject bbox) so the crops can be labeled on the existing 0-4 grade +
     ``FailureType`` schema and matched to originals later.

This does NOT modify the database or the source NEFs. It only reads.

Connection defaults match docker-compose.yml (host-mapped Postgres):
  host=127.0.0.1 port=5432 db=image_scoring user=postgres password=postgres
Override via --pg-* flags or POSTGRES_* env vars.

Run with the Store Python 3.11 that has torch/ultralytics/eye_quality:
  & "C:\\Users\\dmnsy\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe" \\
      training/build_eye_dataset.py --limit 500 --max-per-species 40
"""

from __future__ import annotations

import os

# Store Python's site-packages is read-only, so Ultralytics' auto-install of
# optional deps (e.g. pi-heif) fails noisily on a few files. Disable it; the
# embedded NEF previews are plain JPEG and need no extra codecs.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import argparse
import io
import json
import re
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from eye_quality.crop.eye_crop import OrientedImage, load_oriented_image
from eye_quality.localization.pose_model import PoseLocalizer, resolve_weights_path

RAW_EXTS = {".nef", ".nrw", ".arw", ".cr2", ".cr3", ".dng", ".orf", ".rw2"}


# ---------------------------------------------------------------------------
# Path translation
# ---------------------------------------------------------------------------

def wsl_to_win(path: str) -> str:
    """Translate a stored WSL mount path to a Windows path.

    ``/mnt/d/Photos/x`` -> ``D:\\Photos\\x``. Returns the input unchanged if it
    is already a native path.
    """
    m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", path)
    if m:
        return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\")
    return path


# ---------------------------------------------------------------------------
# Full-resolution NEF preview extraction (dependency-free)
# ---------------------------------------------------------------------------

def _extract_from_subifd(data: bytes) -> bytes | None:
    """Parse the TIFF SubIFD chain and return the largest embedded JPEG."""
    try:
        if data[0:2] == b"II":
            endian = "<"
        elif data[0:2] == b"MM":
            endian = ">"
        else:
            return None
        if struct.unpack(f"{endian}H", data[2:4])[0] != 42:
            return None
        ifd0 = struct.unpack(f"{endian}I", data[4:8])[0]

        sub_offsets: list[int] = []
        n = struct.unpack(f"{endian}H", data[ifd0:ifd0 + 2])[0]
        for i in range(n):
            e = ifd0 + 2 + i * 12
            tag = struct.unpack(f"{endian}H", data[e:e + 2])[0]
            if tag == 0x014A:  # SubIFDs
                sub_offsets.append(struct.unpack(f"{endian}I", data[e + 8:e + 12])[0])

        best: tuple[int, int] | None = None
        for so in sub_offsets:
            n2 = struct.unpack(f"{endian}H", data[so:so + 2])[0]
            off = length = None
            for i in range(n2):
                e = so + 2 + i * 12
                tag = struct.unpack(f"{endian}H", data[e:e + 2])[0]
                val = struct.unpack(f"{endian}I", data[e + 8:e + 12])[0]
                if tag == 0x0201:  # JPEGInterchangeFormat (offset)
                    off = val
                elif tag == 0x0202:  # JPEGInterchangeFormatLength
                    length = val
            if (
                off and length
                and off + length <= len(data)
                and data[off:off + 2] == b"\xff\xd8"
            ):
                if best is None or length > best[1]:
                    best = (off, length)
        if best:
            return data[best[0]:best[0] + best[1]]
        return None
    except Exception:
        return None


def _extract_by_marker_scan(data: bytes) -> bytes | None:
    """Scan for SOI/EOI markers and return the largest embedded JPEG."""
    try:
        sois = [
            i for i in range(512, len(data) - 1)
            if data[i] == 0xFF and data[i + 1] == 0xD8
        ]
        best: tuple[int, int] | None = None
        for s in sois:
            for i in range(s + 2, len(data) - 1):
                if data[i] == 0xFF and data[i + 1] == 0xD9:
                    size = i + 2 - s
                    if size > 10000 and (best is None or size > best[1]):
                        best = (s, size)
                    break
        if best:
            return data[best[0]:best[0] + best[1]]
        return None
    except Exception:
        return None


def extract_full_preview(raw_path: Path) -> tuple[bytes, str] | None:
    """Return (jpeg_bytes, method) of the largest embedded preview, or None."""
    try:
        data = raw_path.read_bytes()
    except OSError:
        return None
    jpg = _extract_from_subifd(data)
    method = "subifd"
    if jpg is None:
        jpg = _extract_by_marker_scan(data)
        method = "marker_scan"
    if jpg is None:
        return None
    return jpg, method


def crop_eye_for_dataset(
    oriented: OrientedImage,
    center_x: float,
    center_y: float,
    head_width_px: float,
    out_path: Path,
    *,
    crop_scale: float,
    min_crop_px: int,
) -> tuple[int, int, list[int]] | None:
    """Crop a square eye region with labeling context and save it.

    Unlike the production ``crop_eye_region`` (radius = 0.12*head_width, 24px
    floor — tuned for the heuristic scorer), this captures the eye plus orbital
    feathers so a human/classifier can judge focus. Side = ``crop_scale`` *
    head_width, clamped to a higher ``min_crop_px`` floor. Pulls from the native
    full-res array; no upscaling. Returns (w, h, [x1,y1,x2,y2]) or None if empty.
    """
    side = max(head_width_px * crop_scale, float(min_crop_px))
    half = side / 2.0
    cx, cy = int(round(center_x)), int(round(center_y))
    x1 = max(0, cx - int(round(half)))
    y1 = max(0, cy - int(round(half)))
    x2 = min(oriented.width, cx + int(round(half)))
    y2 = min(oriented.height, cy + int(round(half)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop_rgb = oriented.array_rgb[y1:y2, x1:x2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop_rgb).save(out_path, quality=95)
    return crop_rgb.shape[1], crop_rgb.shape[0], [x1, y1, x2, y2]


def oriented_from_jpeg_bytes(jpeg: bytes, source_path: str) -> OrientedImage:
    """Build an EXIF-oriented OrientedImage from in-memory JPEG bytes."""
    import numpy as np
    from PIL import ImageOps

    with Image.open(io.BytesIO(jpeg)) as img:
        oriented = ImageOps.exif_transpose(img)
        arr = np.array(oriented.convert("RGB"))
    h, w = arr.shape[:2]
    return OrientedImage(array_rgb=arr, width=w, height=h, source_path=source_path)


# ---------------------------------------------------------------------------
# Database query
# ---------------------------------------------------------------------------

def fetch_species_images(args) -> list[dict]:
    """Return rows of species-labeled images: id, file_path, file_name, species, confidence."""
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(
        host=args.pg_host,
        port=args.pg_port,
        dbname=args.pg_db,
        user=args.pg_user,
        password=args.pg_password,
    )
    # NB: literal % is doubled (%%) because psycopg2 applies %-formatting whenever
    # a (non-None) params sequence is passed, even an empty one.
    where = ["kd.keyword_norm LIKE 'species:%%'"]
    params: list = []
    if args.raw_only:
        where.append("LOWER(i.file_name) LIKE '%%.nef'")
    if args.species:
        where.append("LOWER(kd.keyword_display) = LOWER(%s)")
        params.append(f"species:{args.species}")
    if args.min_species_conf > 0:
        where.append("(ik.confidence IS NULL OR ik.confidence >= %s)")
        params.append(args.min_species_conf)

    # One row per (image, species). DISTINCT ON keeps the highest-confidence
    # species per image so an image labeled with multiple species isn't duplicated.
    sql = f"""
        SELECT DISTINCT ON (i.id)
               i.id, i.file_path, i.file_name, i.burst_uuid,
               kd.keyword_display AS species_kw, ik.confidence
        FROM image_keywords ik
        JOIN keywords_dim kd ON ik.keyword_id = kd.keyword_id
        JOIN images i ON i.id = ik.image_id
        WHERE {" AND ".join(where)}
        ORDER BY i.id, ik.confidence DESC NULLS LAST
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    for r in rows:
        kw = (r.pop("species_kw") or "").strip()
        r["species"] = kw[len("species:"):] if kw.lower().startswith("species:") else kw
    return rows


def balance_and_limit(rows: list[dict], args) -> list[dict]:
    """Order, then apply per-burst dedup, per-species cap, global offset, and limit.

    Default order is a seeded shuffle so the sample is representative of the whole
    library rather than front-loading the oldest (lowest-id) imports, which in this
    DB are dirty test data (distant/soft birds, non-bird BioCLIP false positives).
    """
    if args.order == "asc":
        rows = sorted(rows, key=lambda r: r["id"])
    elif args.order == "desc":
        rows = sorted(rows, key=lambda r: r["id"], reverse=True)
    else:  # random (default)
        import random
        rows = sorted(rows, key=lambda r: r["id"])  # stable base before shuffle
        random.Random(args.seed).shuffle(rows)
    if args.max_per_burst > 0:
        # Cap frames per burst so a long shutter burst doesn't flood the dataset
        # with near-duplicates. Images without a burst_uuid are their own burst.
        per_burst: Counter = Counter()
        kept = []
        for r in rows:
            key = r.get("burst_uuid") or f"_single_{r['id']}"
            if per_burst[key] >= args.max_per_burst:
                continue
            per_burst[key] += 1
            kept.append(r)
        rows = kept
    if args.max_per_species > 0:
        per: Counter[str] = Counter()
        kept = []
        for r in rows:
            sp = r["species"]
            if per[sp] >= args.max_per_species:
                continue
            per[sp] += 1
            kept.append(r)
        rows = kept
    if args.offset:
        rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Build eye-crop dataset from species-labeled NEFs")
    # DB
    p.add_argument("--pg-host", default=os.environ.get("POSTGRES_HOST", "127.0.0.1"))
    p.add_argument("--pg-port", type=int, default=int(os.environ.get("POSTGRES_PORT", 5432)))
    p.add_argument("--pg-db", default=os.environ.get("POSTGRES_DB", "image_scoring"))
    p.add_argument("--pg-user", default=os.environ.get("POSTGRES_USER", "postgres"))
    p.add_argument("--pg-password", default=os.environ.get("POSTGRES_PASSWORD", "postgres"))
    # Selection
    p.add_argument("--species", default=None, help="Only this species (common name, e.g. 'Snowy Egret')")
    p.add_argument("--min-species-conf", type=float, default=0.0, help="Min BioCLIP species confidence")
    p.add_argument("--max-per-species", type=int, default=0, help="Cap crops-source images per species (0=all)")
    p.add_argument("--max-per-burst", type=int, default=2, help="Cap frames per shutter burst to limit near-duplicates (0=all)")
    p.add_argument("--limit", type=int, default=0, help="Max source images to process (0=all)")
    p.add_argument("--offset", type=int, default=0, help="Skip the first N images (resume/sharding)")
    p.add_argument("--order", choices=("random", "asc", "desc"), default="random",
                   help="Sampling order: random (seeded, representative), asc/desc by image id")
    p.add_argument("--seed", type=int, default=13, help="Shuffle seed for --order random")
    p.add_argument("--min-subject-frac", type=float, default=0.01,
                   help="Drop birds smaller than this fraction of frame area (eye too few px)")
    p.add_argument("--min-head-width-px", type=float, default=200.0,
                   help="Drop eyes whose estimated head width is below this (eye ~0.25*head; "
                        "<~200px head means <~50px eye, too small to grade for sharpness)")
    p.add_argument("--raw-only", action="store_true", default=True, help="Restrict to .NEF (default on)")
    p.add_argument("--no-raw-only", dest="raw_only", action="store_false")
    # Localization
    p.add_argument("--weights", default=None, help="Pose weights (default: resolve_weights_path)")
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=1280, help="Inference size (higher finds small/distant birds)")
    p.add_argument("--min-eye-conf", type=float, default=0.5, help="Drop eyes below this keypoint conf (hidden/far eyes)")
    p.add_argument(
        "--visibility",
        default="visible,partially_visible,occluded",
        help="Comma list of eye visibilities to keep",
    )
    p.add_argument(
        "--crop-scale",
        type=float,
        default=0.6,
        help="Eye-crop side as a fraction of estimated head width (more context than production)",
    )
    p.add_argument(
        "--min-crop-px",
        type=int,
        default=64,
        help="Minimum eye-crop side in native px (no upscaling beyond available pixels)",
    )
    # Output
    p.add_argument("--output-dir", default="data/eye_dataset")
    args = p.parse_args()

    keep_vis = {v.strip() for v in args.visibility.split(",") if v.strip()}

    out_dir = Path(args.output_dir).resolve()
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    summary_path = out_dir / "summary.json"

    print(f"Querying Postgres {args.pg_host}:{args.pg_port}/{args.pg_db} for species-labeled images...")
    rows = fetch_species_images(args)
    print(f"  {len(rows)} species-labeled source images matched filters.")
    rows = balance_and_limit(rows, args)
    print(f"  Processing {len(rows)} after offset/limit/per-species cap.")
    if not rows:
        raise SystemExit("No images to process.")

    weights = resolve_weights_path(args.weights)
    localizer = PoseLocalizer(weights=weights, device=args.device, imgsz=args.imgsz)
    print(f"Localizer weights: {weights} (imgsz={args.imgsz}, device={args.device or 'auto'})")

    stats = Counter()
    species_crops: Counter[str] = Counter()
    crops_written = 0
    tmp_preview = out_dir / "_preview_tmp.jpg"

    with manifest_path.open("w", encoding="utf-8") as mf:
        for n, row in enumerate(rows, 1):
            stats["images_seen"] += 1
            wsl_path = row["file_path"] or ""
            win_path = wsl_to_win(wsl_path)
            src = Path(win_path)
            ext = src.suffix.lower()

            if not src.is_file():
                stats["missing_file"] += 1
                continue

            # Build a native-resolution oriented image.
            preview_method = None
            try:
                if ext in RAW_EXTS:
                    res = extract_full_preview(src)
                    if res is None:
                        stats["preview_extract_failed"] += 1
                        continue
                    jpeg, preview_method = res
                    oriented = oriented_from_jpeg_bytes(jpeg, str(src))
                else:
                    oriented = load_oriented_image(src)
                    preview_method = "native"
            except Exception as exc:  # noqa: BLE001 - keep batch robust
                stats["decode_error"] += 1
                if stats["decode_error"] <= 5:
                    print(f"  [decode error] {src.name}: {exc}")
                continue

            # Normalize the oriented frame to a temp JPEG with orientation baked
            # in and EXIF stripped. PoseLocalizer.predict() re-opens the path
            # without applying EXIF transpose, so feeding it the already-oriented
            # pixels guarantees keypoint coords line up with the array we crop
            # from below (and lets it read NEFs, which PIL/YOLO cannot open raw).
            try:
                Image.fromarray(oriented.array_rgb).save(tmp_preview, quality=95)
            except Exception as exc:  # noqa: BLE001
                stats["preview_write_error"] += 1
                continue

            # Localize eyes on the full-res preview.
            try:
                localizations, _kpts, notes = localizer.localize_eyes(
                    tmp_preview, oriented.width, oriented.height
                )
            except Exception as exc:  # noqa: BLE001
                stats["localize_error"] += 1
                if stats["localize_error"] <= 5:
                    print(f"  [localize error] {src.name}: {exc}")
                continue

            if "no_pose_detection" in notes:
                stats["no_pose_detection"] += 1
                continue
            if not localizations:
                stats["no_eyes"] += 1
                continue
            stats["images_with_detection"] += 1

            # Drop distant/small subjects: if the bird occupies too little of the
            # frame, even a native-res crop has too few eye pixels to be legible.
            if (
                args.min_subject_frac > 0
                and localizations[0].subject_area_frac < args.min_subject_frac
            ):
                stats["subject_too_small"] += 1
                continue

            kept_any = False
            for loc in localizations:
                if loc.confidence < args.min_eye_conf:
                    stats["eye_below_conf"] += 1
                    continue
                if loc.visibility.value not in keep_vis:
                    stats[f"vis_skip_{loc.visibility.value}"] += 1
                    continue
                if loc.head_width_px < args.min_head_width_px:
                    stats["head_too_small"] += 1
                    continue

                eye_side = loc.eye_side.value if hasattr(loc.eye_side, "value") else str(loc.eye_side)
                crop_path = crops_dir / f"img{row['id']}_{eye_side}_eye.jpg"
                crop = crop_eye_for_dataset(
                    oriented,
                    loc.center_x,
                    loc.center_y,
                    loc.head_width_px,
                    crop_path,
                    crop_scale=args.crop_scale,
                    min_crop_px=args.min_crop_px,
                )
                if crop is None:
                    stats["crop_empty"] += 1
                    continue
                crop_w, crop_h, eye_box_px = crop

                rec = {
                    "crop_path": os.path.relpath(crop_path, out_dir).replace("\\", "/"),
                    "image_id": int(row["id"]),
                    "species": row["species"],
                    "species_confidence": (
                        float(row["confidence"]) if row["confidence"] is not None else None
                    ),
                    "source_path_win": str(src),
                    "source_path_wsl": wsl_path,
                    "file_name": row["file_name"],
                    "burst_uuid": row.get("burst_uuid"),
                    "eye_side": eye_side,
                    "visibility": loc.visibility.value,
                    "eye_confidence": round(float(loc.confidence), 4),
                    "center_x": round(float(loc.center_x), 2),
                    "center_y": round(float(loc.center_y), 2),
                    "head_width_px": round(float(loc.head_width_px), 2),
                    "crop_width_px": crop_w,
                    "crop_height_px": crop_h,
                    "eye_box_px": eye_box_px,  # [x1,y1,x2,y2] of the crop in the preview
                    "subject_area_frac": round(float(loc.subject_area_frac), 5),
                    "eye_bbox_norm": [round(float(v), 5) for v in loc.bbox_norm],
                    "preview_width": oriented.width,
                    "preview_height": oriented.height,
                    "preview_method": preview_method,
                    # Labeling targets — filled in by the labeling pass:
                    "grade": None,        # 0-4 eye-quality grade
                    "failure_type": None, # FailureType enum value
                }
                mf.write(json.dumps(rec) + "\n")
                crops_written += 1
                species_crops[row["species"]] += 1
                kept_any = True

            if kept_any:
                stats["images_with_crops"] += 1
            if n % 50 == 0 or n == len(rows):
                print(
                    f"  [{n}/{len(rows)}] crops={crops_written} "
                    f"det={stats['images_with_detection']} "
                    f"miss={stats['missing_file']} nopose={stats['no_pose_detection']}",
                    flush=True,
                )

    try:
        tmp_preview.unlink()
    except OSError:
        pass

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weights": weights,
        "imgsz": args.imgsz,
        "order": args.order,
        "seed": args.seed,
        "min_eye_conf": args.min_eye_conf,
        "min_subject_frac": args.min_subject_frac,
        "min_head_width_px": args.min_head_width_px,
        "crop_scale": args.crop_scale,
        "min_crop_px": args.min_crop_px,
        "kept_visibilities": sorted(keep_vis),
        "source_images_processed": stats["images_seen"],
        "images_with_detection": stats["images_with_detection"],
        "images_with_crops": stats["images_with_crops"],
        "crops_written": crops_written,
        "manifest": str(manifest_path),
        "crops_dir": str(crops_dir),
        "diagnostics": dict(stats),
        "crops_per_species": dict(species_crops.most_common()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Done ===")
    print(json.dumps({k: summary[k] for k in (
        "crops_written", "images_with_crops", "images_with_detection",
        "source_images_processed")}, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
