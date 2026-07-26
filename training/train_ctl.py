#!/usr/bin/env python3
"""Pause/resume helpers for YOLO pose and detect training runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = REPO_ROOT / "runs" / ".train_state.json"

DEFAULT_PYTHON = os.environ.get(
    "EYE_QUALITY_PYTHON",
    r"C:\Users\dmnsy\AppData\Local\Microsoft\WindowsApps\python.exe"
    if sys.platform == "win32"
    else sys.executable,
)

TASKS: dict[str, dict[str, str]] = {
    "detect": {
        "script": "training/train_detect.py",
        "project": "runs/detect",
        "name": "wildlife_bird",
        "output": "models/bird_detect_v0.pt",
        "data": "training/configs/wildlife_bird_det.yaml",
        "base": "yolo11n.pt",
    },
    "pose": {
        "script": "training/train_pose.py",
        "project": "runs/pose",
        "name": "wildlife_bird",
        "output": "models/eye_pose_v0.pt",
        "data": "training/configs/wildlife_bird.yaml",
        "base": "yolo11n-pose.pt",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _find_pids(script_name: str) -> list[int]:
    """Find training PIDs by script filename (e.g. train_detect.py)."""
    if sys.platform == "win32":
        ps = (
            "Get-CimInstance Win32_Process | "
            f"Where-Object {{ $_.CommandLine -like '*{script_name}*' }} | "
            "ForEach-Object { $_.ProcessId }"
        )
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            check=False,
        )
        pids: list[int] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids

    proc = subprocess.run(["pgrep", "-f", script_name], capture_output=True, text=True, check=False)
    return [int(line) for line in proc.stdout.splitlines() if line.strip().isdigit()]


def _terminate_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        if not _pid_alive(pid):
            return
        time.sleep(0.5)
    os.kill(pid, signal.SIGKILL)


def _save_dir(task: str) -> Path:
    meta = TASKS[task]
    return REPO_ROOT / meta["project"] / meta["name"]


def _checkpoint(task: str) -> Path:
    return _save_dir(task) / "weights" / "last.pt"


def _read_progress(task: str) -> dict[str, Any]:
    save_dir = _save_dir(task)
    results_csv = save_dir / "results.csv"
    progress: dict[str, Any] = {
        "save_dir": str(save_dir),
        "checkpoint": str(_checkpoint(task)),
        "checkpoint_exists": _checkpoint(task).is_file(),
    }
    if results_csv.is_file():
        with results_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            last = rows[-1]
            progress["completed_epochs"] = int(float(last.get("epoch", 0)))
            progress["last_metrics"] = last
    return progress


def _build_train_command(
    task: str,
    *,
    python_exe: str,
    resume: bool,
    epochs: int,
    batch: int,
    device: str | None,
    workers: int,
    cache: str,
    output: str,
    base: str | None = None,
) -> list[str]:
    meta = TASKS[task]
    cmd = [
        python_exe,
        str(REPO_ROOT / meta["script"]),
        "--data",
        meta["data"],
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--cache",
        cache,
        "--output",
        output,
    ]
    if device:
        cmd.extend(["--device", device])
    if resume:
        ckpt = _checkpoint(task)
        if not ckpt.is_file():
            raise SystemExit(f"Cannot resume {task}: checkpoint not found at {ckpt}")
        cmd.extend(["--resume", str(ckpt)])
    else:
        cmd.extend(["--base", base or meta["base"]])
    return cmd


def cmd_status(args: argparse.Namespace) -> None:
    state = _load_state()
    task = args.task or state.get("task")
    if not task:
        print("No active or recorded training task.")
        return

    progress = _read_progress(task)
    pid = state.get("pid") if state.get("task") == task else None
    running = bool(pid and _pid_alive(int(pid)))
    if not running:
        script = Path(TASKS[task]["script"]).name
        discovered = _find_pids(script)
        if discovered:
            pid = discovered[0]
            running = True

    print(f"task: {task}")
    print(f"status: {'running' if running else 'stopped'}")
    if pid:
        print(f"pid: {pid}")
    if state.get("task") == task:
        print(f"state: {state.get('status', 'unknown')}")
        if state.get("command"):
            print(f"command: {' '.join(state['command'])}")
    print(f"save_dir: {progress['save_dir']}")
    print(f"checkpoint: {progress['checkpoint']} ({'present' if progress['checkpoint_exists'] else 'missing'})")
    if "completed_epochs" in progress:
        print(f"completed_epochs: {progress['completed_epochs']}")
    else:
        print("completed_epochs: 0 (no results.csv yet)")


def cmd_pause(args: argparse.Namespace) -> None:
    state = _load_state()
    task = args.task or state.get("task")
    if not task:
        raise SystemExit("No task specified and no saved training state found.")

    script = Path(TASKS[task]["script"]).name
    pids = _find_pids(script)
    if not pids and state.get("pid"):
        pids = [int(state["pid"])]

    if not pids:
        state["status"] = "stopped"
        state["paused_at"] = _utc_now()
        _save_state(state)
        print(f"No running {task} training process found.")
        return

    for pid in pids:
        print(f"Pausing {task} training (pid {pid})...")
        _terminate_pid(pid)

    state.update(
        {
            "task": task,
            "status": "paused",
            "pid": None,
            "paused_at": _utc_now(),
            "resume_checkpoint": str(_checkpoint(task)),
        }
    )
    _save_state(state)
    print(f"Paused. Resume with: python -m training.train_ctl resume --task {task}")


def cmd_resume(args: argparse.Namespace) -> None:
    state = _load_state()
    task = args.task or state.get("task")
    if not task:
        raise SystemExit("No task specified and no saved training state found.")

    epochs = args.epochs or state.get("epochs", 100)
    batch = args.batch or state.get("batch", 16)
    device = args.device if args.device is not None else state.get("device", "0")
    workers = args.workers if args.workers is not None else state.get("workers", 2 if sys.platform == "win32" else 8)
    cache = args.cache or state.get("cache", "disk")
    output = args.output or state.get("output", TASKS[task]["output"])
    python_exe = args.python or state.get("python", DEFAULT_PYTHON)

    cmd = _build_train_command(
        task,
        python_exe=python_exe,
        resume=True,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        cache=cache,
        output=output,
    )

    if args.background:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        pid = proc.pid
    else:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
        pid = proc.pid
        proc.wait()
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    state.update(
        {
            "task": task,
            "status": "running",
            "pid": pid,
            "started_at": _utc_now(),
            "command": cmd,
            "epochs": epochs,
            "batch": batch,
            "device": device,
            "workers": workers,
            "cache": cache,
            "output": output,
            "python": python_exe,
            "resume_checkpoint": str(_checkpoint(task)),
        }
    )
    _save_state(state)
    print(f"Resumed {task} training (pid {pid}).")
    print(f"Command: {' '.join(cmd)}")


def cmd_start(args: argparse.Namespace) -> None:
    task = args.task
    epochs = args.epochs
    batch = args.batch
    device = args.device
    workers = args.workers if args.workers is not None else (2 if sys.platform == "win32" else 8)
    cache = args.cache
    output = args.output or TASKS[task]["output"]
    python_exe = args.python or DEFAULT_PYTHON
    resume = args.resume or _checkpoint(task).is_file()

    cmd = _build_train_command(
        task,
        python_exe=python_exe,
        resume=resume,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        cache=cache,
        output=output,
        base=args.base,
    )

    if args.background:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        pid = proc.pid
    else:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
        pid = proc.pid
        proc.wait()
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    state = {
        "task": task,
        "status": "running",
        "pid": pid,
        "started_at": _utc_now(),
        "command": cmd,
        "epochs": epochs,
        "batch": batch,
        "device": device,
        "workers": workers,
        "cache": cache,
        "output": output,
        "python": python_exe,
        "resume_checkpoint": str(_checkpoint(task)),
    }
    _save_state(state)
    mode = "Resumed" if resume else "Started"
    print(f"{mode} {task} training (pid {pid}).")
    print(f"Command: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pause/resume YOLO training runs")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", choices=sorted(TASKS), help="Training task (pose or detect)")

    p_status = sub.add_parser("status", parents=[common], help="Show training status")
    p_status.set_defaults(func=cmd_status)

    p_pause = sub.add_parser("pause", parents=[common], help="Stop the running training process")
    p_pause.set_defaults(func=cmd_pause)

    train_args = argparse.ArgumentParser(add_help=False)
    train_args.add_argument("--epochs", type=int, default=100)
    train_args.add_argument("--batch", type=int, default=16)
    train_args.add_argument("--device", default="0")
    train_args.add_argument("--workers", type=int, default=None)
    train_args.add_argument("--cache", choices=("none", "ram", "disk"), default="disk")
    train_args.add_argument("--output", default=None)
    train_args.add_argument("--python", default=None, help="Python executable with torch/CUDA")
    train_args.add_argument("--background", action="store_true", help="Run detached in background")

    p_start = sub.add_parser("start", parents=[common, train_args], help="Start a training run")
    p_start.add_argument("--base", default=None, help="Base weights when not resuming")
    p_start.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last.pt if present (default when checkpoint exists)",
    )
    p_start.set_defaults(func=cmd_start)

    p_resume = sub.add_parser("resume", parents=[common, train_args], help="Resume from last.pt")
    p_resume.set_defaults(func=cmd_resume)

    args = parser.parse_args()
    if args.command in ("start",) and not getattr(args, "task", None):
        parser.error("start requires --task pose|detect")
    args.func(args)


if __name__ == "__main__":
    main()
