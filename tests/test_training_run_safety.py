"""New training runs must not overwrite checkpoints or prior metrics."""

from __future__ import annotations

import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training.run_safety import prepare_run_paths
from training import train_ctl


class RunSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_new_run_uses_candidate_output_and_fails_if_repeated(self) -> None:
        run_dir, output = prepare_run_paths(self.root, "detect", "trial_1", None)
        self.assertEqual(run_dir, self.root / "runs" / "detect" / "trial_1")
        self.assertEqual(output, self.root / "models" / "candidates" / "detect_trial_1.pt")
        run_dir.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_run_paths(self.root, "detect", "trial_1", None)

    def test_rejects_released_output_even_when_not_present(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate"):
            prepare_run_paths(self.root, "pose", "trial_1", "models/eye_pose_v0.pt")

    def test_rejects_output_that_would_replace_run_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate"):
            prepare_run_paths(self.root, "pose", "trial_1", "runs/pose/trial_1/weights/last.pt")

    def test_rejects_existing_candidate_output_for_new_run(self) -> None:
        output = self.root / "models" / "candidates" / "pose_trial_1.pt"
        output.parent.mkdir(parents=True)
        output.touch()
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_run_paths(self.root, "pose", "trial_1", output)

    def test_resume_requires_checkpoint_in_named_run(self) -> None:
        run_dir = self.root / "runs" / "pose" / "trial_1"
        checkpoint = run_dir / "weights" / "last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        actual_run, output = prepare_run_paths(self.root, "pose", "trial_1", None, resume=checkpoint)
        self.assertEqual(actual_run, run_dir)
        self.assertEqual(output, self.root / "models" / "candidates" / "pose_trial_1.pt")
        with self.assertRaisesRegex(ValueError, "named run"):
            prepare_run_paths(self.root, "pose", "trial_2", None, resume=checkpoint)

    def test_rejects_name_as_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "name"):
            prepare_run_paths(self.root, "detect", "../released", None)

    def test_controller_rejects_released_output_before_spawning(self) -> None:
        args = SimpleNamespace(task="detect", epochs=1, batch=1, device="0", workers=0,
                               cache="none", output="models/bird_detect_v0.pt", python=None,
                               resume=False, base=None, background=False)
        with patch.object(train_ctl, "_spawn_training", side_effect=AssertionError("must not spawn")):
            with self.assertRaisesRegex(ValueError, "candidate"):
                train_ctl.cmd_start(args)

    def test_direct_train_scripts_reach_output_guard(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for script, released in (("train_detect.py", "bird_detect_v0.pt"),
                                 ("train_pose.py", "eye_pose_v0.pt")):
            with self.subTest(script=script):
                proc = subprocess.run(
                    [sys.executable, str(repo_root / "training" / script), "--name", "guard_probe",
                     "--output", f"models/{released}", "--epochs", "1"],
                    cwd=repo_root, capture_output=True, text=True, check=False,
                )
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("candidate", proc.stderr)
                self.assertNotIn("ModuleNotFoundError", proc.stderr)


if __name__ == "__main__":
    unittest.main()
