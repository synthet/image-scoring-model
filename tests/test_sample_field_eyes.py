"""Unit tests for the pure helpers in training/sample_field_eyes.py (no models, no photo library)."""
from __future__ import annotations

import json

from training.sample_field_eyes import excluded_by, size_stratum, year_of


def test_excluded_by_collects_bird_folders_and_all_sources(tmp_path):
    rows = [
        {"kind": "bird", "group_id": "Z8/400mm/2025/2025-05-01", "source_path": "L:/Z8/400mm/2025/2025-05-01/A.NEF"},
        {"kind": "verified_negative", "group_id": "Z8/x/2024", "source_path": "L:/Z8/x/2024/B.NEF"},
    ]
    m = tmp_path / "manifest.jsonl"
    m.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n", encoding="utf-8")
    groups, sources = excluded_by([str(m)])
    # a negative's folder is not excluded (it was drawn from a list, not from the folder walk)
    assert groups == {"Z8/400mm/2025/2025-05-01"}
    assert len(sources) == 2 and all(s == s.lower() for s in sources)
    assert excluded_by([]) == (set(), set())


def test_size_strata_and_year():
    assert [size_stratum(a) for a in (0.01, 0.04, 0.149, 0.15)] == ["small", "medium", "medium", "large"]
    assert year_of(("Z6ii", "28-400mm", "2025", "2025-07-01")) == "2025"
    assert year_of(("Camera JPEG", "misc")) == "unknown"
