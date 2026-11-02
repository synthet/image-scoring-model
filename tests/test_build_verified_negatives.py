"""Owner-reviewed bird-free frames can be imported without label leakage."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from training.build_verified_negatives import build_manifest, resolve_photo_path


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


class VerifiedNegativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "Photos"
        self.root.mkdir()
        self.source = self.root / "D90" / "2022-12-10" / "birdless.NEF"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"raw-photo-placeholder")
        self.cohort = Path(self.temp.name) / "cohort.csv"
        self.labels = Path(self.temp.name) / "labels.csv"
        write_csv(self.cohort, ["image_id", "file_path", "stratum"], [
            ["1", "/mnt/d/Photos/D90/2022-12-10/birdless.NEF", "negative"],
        ])
        write_csv(self.labels, ["image_id", "label"], [["1", "no_bird"]])

    def test_maps_wsl_photo_path_to_existing_windows_file(self) -> None:
        resolved = resolve_photo_path("/mnt/d/Photos/D90/2022-12-10/birdless.NEF", self.root)
        self.assertEqual(resolved, self.source)

    def test_builds_only_verified_negatives_with_source_hash(self) -> None:
        records = build_manifest(self.cohort, self.labels, self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["image_id"], "1")
        self.assertEqual(records[0]["label"], "human_verified_no_bird")
        self.assertEqual(records[0]["group_id"], "D90/2022-12-10")
        self.assertEqual(len(records[0]["source_sha256"]), 64)

    def test_rejects_path_escaping_photo_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "photo root"):
            resolve_photo_path("/mnt/d/Photos/../outside.NEF", self.root)

    def test_rejects_training_folder_in_holdout(self) -> None:
        holdout = Path(self.temp.name) / "holdout.csv"
        write_csv(holdout, ["image_id", "file_path", "stratum"], [
            ["2", "/mnt/d/Photos/D90/2022-12-10/other.NEF", "test"],
        ])
        with self.assertRaisesRegex(ValueError, "holdout"):
            build_manifest(self.cohort, self.labels, self.root, holdout_cohort=holdout)

    def test_rejects_unreviewed_or_duplicate_owner_labels(self) -> None:
        write_csv(self.labels, ["image_id", "label"], [["1", "unsure"]])
        self.assertEqual(build_manifest(self.cohort, self.labels, self.root), [])
        write_csv(self.labels, ["image_id", "label"], [["1", "no_bird"], ["1", "bird"]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_manifest(self.cohort, self.labels, self.root)


if __name__ == "__main__":
    unittest.main()
