from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "analysis" / "analyze_project_genie_turn.py"
SPEC = importlib.util.spec_from_file_location("analyze_project_genie_turn", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sample_path_falls_back_to_portable_jpeg(tmp_path: Path) -> None:
    samples = tmp_path / "turn_samples.json"
    samples.write_text("[]\n", encoding="utf-8")
    portable = tmp_path / "frame_0007.jpg"
    portable.write_bytes(b"jpeg")
    row = {"index": 7, "path": "/unavailable/original/frame.png"}
    assert MODULE._sample_path(samples, row) == portable


def test_sample_path_prefers_recorded_source(tmp_path: Path) -> None:
    samples = tmp_path / "turn_samples.json"
    samples.write_text("[]\n", encoding="utf-8")
    recorded = tmp_path / "original.png"
    recorded.write_bytes(b"png")
    portable = tmp_path / "frame_0001.jpg"
    portable.write_bytes(b"jpeg")
    row = {"index": 1, "path": str(recorded)}
    assert MODULE._sample_path(samples, row) == recorded
