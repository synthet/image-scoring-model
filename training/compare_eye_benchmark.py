"""Paired comparison of two saved eye-benchmark runs on the same labelled sample.

Each bootstrap draw resamples frames once and scores both runs on those frames, so the interval is for the
difference itself. Reads saved predictions only: no weights, no model, nothing written.

    python -m training.compare_eye_benchmark --sample data/field_eye_v1 \
        --base data/eye_benchmark/d1_test_base --candidate data/eye_benchmark/d1_test_cand
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from training.eval_eye_benchmark import aggregate, field_frames, frame_records

KEYS = ("negative_frames_with_false_eye", "bird_recall", "pck@0.05", "frame_eye_hit_rate", "false_eyes_on_hidden_rate")


def paired_difference(base: list[dict], cand: list[dict], key: str, n: int = 2000,
                      seed: int = 0) -> tuple[float, float, float]:
    """Candidate minus base for ``key``, with a 95% percentile interval. Draws where it is undefined are skipped."""
    rng, size, diffs = random.Random(seed), len(base), []
    for _ in range(n):
        idx = [rng.randrange(size) for _ in range(size)]
        b, c = aggregate([base[i] for i in idx])[key], aggregate([cand[i] for i in idx])[key]
        if b is not None and c is not None:
            diffs.append(c - b)
    diffs.sort()
    return aggregate(cand)[key] - aggregate(base)[key], diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def records(frames: list[dict], folder: Path, model: str, floor: float, bird_conf: float) -> list[dict]:
    with open(folder / f"{model}.predictions.jsonl", encoding="utf-8") as src:
        preds = {r["image_id"]: r for r in map(json.loads, src)}
    return [frame_records(f, preds.get(f["image_id"], {}), floor, bird_conf) for f in frames]


def compare(frames: list[dict], base_dir: Path, cand_dir: Path, model: str, floor: float, bird_conf: float,
            keys: tuple[str, ...] = KEYS, n: int = 2000) -> list[dict]:
    base, cand = records(frames, base_dir, model, floor, bird_conf), records(frames, cand_dir, model, floor, bird_conf)
    base_agg, cand_agg = aggregate(base), aggregate(cand)
    rows = []
    for key in keys:
        diff, low, high = paired_difference(base, cand, key, n)
        rows.append({"metric": key, "base": base_agg[key], "candidate": cand_agg[key], "difference": diff,
                     "low": low, "high": high})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired comparison of two saved eye-benchmark runs")
    ap.add_argument("--sample", type=Path, required=True, help="Labelled sample folder both runs scored")
    ap.add_argument("--base", type=Path, required=True, help="Folder with the baseline run's predictions")
    ap.add_argument("--candidate", type=Path, required=True, help="Folder with the candidate run's predictions")
    ap.add_argument("--model", default="P_crop", help="Model name in <model>.predictions.jsonl")
    ap.add_argument("--floor", type=float, default=0.8, help="Consumer eye floor")
    ap.add_argument("--bird-conf", type=float, default=0.25)
    a = ap.parse_args()

    frames = field_frames(a.sample, birds_only=True)
    for r in compare(frames, a.base, a.candidate, a.model, a.floor, a.bird_conf):
        print(f"{r['metric']:32} {r['base']:.3f} -> {r['candidate']:.3f}   "
              f"diff {r['difference']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]")


if __name__ == "__main__":
    main()
