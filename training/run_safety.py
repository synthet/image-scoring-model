"""Path guards shared by bird-detect and eye-pose training entry points."""

from __future__ import annotations

import re
from pathlib import Path

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def candidate_output_path(repo_root: Path, task: str, name: str, output: str | Path | None) -> Path:
    """Resolve an output only inside this experiment's candidate locations."""
    if task not in {"detect", "pose"}:
        raise ValueError(f"invalid task: {task}")
    if not _NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError("run name must be a single safe path component")
    root = Path(repo_root).resolve()
    default_output = (root / "models" / "candidates" / f"{task}_{name}.pt").resolve()
    if output is None:
        out = default_output
    else:
        output_path = Path(output)
        out = (output_path if output_path.is_absolute() else root / output_path).resolve()
    candidate_root = (root / "models" / "candidates").resolve()
    if not out.is_relative_to(candidate_root) or out.suffix != ".pt":
        raise ValueError("output must be a .pt candidate under models/candidates/")
    return out


def prepare_run_paths(
    repo_root: Path, task: str, name: str, output: str | Path | None, *,
    resume: str | Path | None = None,
) -> tuple[Path, Path]:
    """Validate a unique new run or an exact named-run resume before model load."""
    root = Path(repo_root).resolve()
    run_dir = (root / "runs" / task / name).resolve()
    default_output = (root / "models" / "candidates" / f"{task}_{name}.pt").resolve()
    out = candidate_output_path(root, task, name, output)

    if resume is None:
        if run_dir.exists():
            raise ValueError(f"run already exists: {run_dir}; choose a unique --name")
        if out.exists():
            raise ValueError(f"candidate output already exists: {out}")
    else:
        checkpoint = Path(resume).resolve()
        if checkpoint != run_dir / "weights" / "last.pt" or not checkpoint.is_file():
            raise ValueError(f"resume checkpoint must be last.pt in the named run: {run_dir}")
        if out.exists() and out != default_output:
            raise ValueError(f"custom candidate output already exists: {out}")
    return run_dir, out
