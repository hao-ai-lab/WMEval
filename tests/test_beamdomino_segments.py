from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "beamdomino"


def test_detect_segments_produces_four_canonical_events():
    from tools.detect_beamdomino_segments import detect_segments

    with open(FIXTURE_DIR / "tiny_log.json") as f:
        log = json.load(f)
    out = detect_segments(log, plateau_window_sec=0.5)

    with open(FIXTURE_DIR / "tiny_segments.json") as f:
        expected = json.load(f)

    assert out["trajectory_id"] == expected["trajectory_id"]
    assert out["source"] == "unity_gt"
    assert len(out["events"]) == 4
    for got, want in zip(out["events"], expected["events"]):
        assert got["index"] == want["index"]
        assert got["name"] == want["name"]
        assert got["trigger"] == want["trigger"]
        assert abs(got["start_sec"] - want["start_sec"]) < 1e-6
        assert abs(got["end_sec"] - want["end_sec"]) < 1e-6


def test_detect_segments_raises_when_stack_never_hit():
    from tools.detect_beamdomino_segments import detect_segments, SegmentDetectionError

    log = {
        "trajectory_id": "no_hit",
        "fps": 50.0,
        "frames": [
            {"t": 0.0, "stack_hit_flag": 0, "cascade_count": 0.0, "fallen_count": 0, "ball_speed": 0.0},
            {"t": 1.0, "stack_hit_flag": 0, "cascade_count": 0.0, "fallen_count": 0, "ball_speed": 3.0},
        ],
    }
    try:
        detect_segments(log)
    except SegmentDetectionError as e:
        assert "stack_hit_flag" in str(e)
        return
    raise AssertionError("expected SegmentDetectionError")


def test_detect_segments_raises_when_cascade_never_starts():
    from tools.detect_beamdomino_segments import detect_segments, SegmentDetectionError

    log = {
        "trajectory_id": "no_cascade",
        "fps": 50.0,
        "frames": [
            {"t": 0.0, "stack_hit_flag": 0, "cascade_count": 0.0, "fallen_count": 0, "ball_speed": 0.0},
            {"t": 0.5, "stack_hit_flag": 1, "cascade_count": 0.0, "fallen_count": 0, "ball_speed": 3.0},
            {"t": 1.5, "stack_hit_flag": 1, "cascade_count": 0.0, "fallen_count": 0, "ball_speed": 3.0},
        ],
    }
    try:
        detect_segments(log)
    except SegmentDetectionError as e:
        assert "fallen" in str(e).lower()
        return
    raise AssertionError("expected SegmentDetectionError")
