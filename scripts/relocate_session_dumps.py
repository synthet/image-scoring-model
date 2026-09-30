#!/usr/bin/env python3
"""Move CLI session transcripts/dumps into docs/private/session-transcripts/."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

ROOT_PATTERNS = (
    re.compile(r"^codex-session-.*\.md$", re.I),
    re.compile(r".*session-is-being-continued.*\.txt$", re.I),
    re.compile(r"^20\d{2}-\d{2}-\d{2}-.*\.txt$"),
)
SCRATCH_PATTERNS = (
    re.compile(r"^codex-session-.*\.md$", re.I),
    re.compile(r".*session-is-being-continued.*\.txt$", re.I),
    re.compile(r"^20\d{2}-\d{2}-\d{2}-.*\.txt$"),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _matches(name: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(p.match(name) for p in patterns)


def _relocate(src: Path, dest_dir: Path, dry_run: bool) -> str:
    dest = dest_dir / src.name
    if dest.resolve() == src.resolve():
        return "skip-same"
    if dest.is_file():
        if _sha256(src) == _sha256(dest):
            if not dry_run:
                src.unlink()
            return "dedupe-removed-src"
        stem, suffix = src.stem, src.suffix
        n = 1
        while dest.is_file():
            dest = dest_dir / f"{stem}.dup{n}{suffix}"
            n += 1
    if dry_run:
        return f"would-move -> {dest.name}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return f"moved -> {dest.name}"


def process_repo(repo: Path, dry_run: bool) -> list[str]:
    dest_dir = repo / "docs" / "private" / "session-transcripts"
    lines: list[str] = []
    candidates: list[Path] = []
    for path in repo.iterdir():
        if path.is_file() and _matches(path.name, ROOT_PATTERNS):
            candidates.append(path)
    scratch = repo / ".agent" / "scratch"
    if scratch.is_dir():
        for path in scratch.rglob("*"):
            if path.is_file() and _matches(path.name, SCRATCH_PATTERNS):
                candidates.append(path)
    for src in sorted(candidates):
        action = _relocate(src, dest_dir, dry_run)
        lines.append(f"  {src.relative_to(repo)}: {action}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repos", nargs="*", type=Path, help="Repo roots (default: image-scoring-* under parent)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repos = args.repos
    if not repos:
        parent = Path(__file__).resolve().parents[1].parent  # D:/Projects
        repos = sorted(parent.glob("image-scoring-*"))
    for repo in repos:
        repo = repo.resolve()
        if not repo.is_dir():
            continue
        print(f"\n=== {repo.name} ===")
        for line in process_repo(repo, args.dry_run):
            print(line)


if __name__ == "__main__":
    main()
