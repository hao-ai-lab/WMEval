from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "beamdomino"


def _vec(x=0.0, y=0.0, z=0.0):
    return {"x": x, "y": y, "z": z}


def _body(name, *, rotation_x=0.0, rotation_z=0.0):
    return {
        "name": name,
        "position": _vec(),
        "velocity": _vec(),
        "angularVelocity": _vec(),
        "rotation": _vec(rotation_x, 0.0, rotation_z),
        "kineticEnergy": 0.0,
        "potentialEnergy": 0.0,
    }


def _collision(a, b, impulse):
    return {
        "time": 0.0,
        "bodyA": a,
        "bodyB": b,
        "contactPoint": _vec(),
        "contactNormal": _vec(),
        "impulse": impulse,
    }


def _frame(idx, t, *, fallen_indices=(), collisions=()):
    bodies = [_body("FloorBall")]
    for di in range(3):
        bodies.append(
            _body(f"Domino_S00_D0{di}", rotation_x=45.0 if di in fallen_indices else 0.0)
        )
    return {
        "frame": idx,
        "time": t,
        "deltaTime": 0.1,
        "bodies": bodies,
        "collisions": list(collisions),
        "totalKineticEnergy": 0.0,
        "totalPotentialEnergy": 0.0,
    }


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


def test_detect_segments_raises_when_no_ball_impact():
    from tools.detect_beamdomino_segments import detect_segments, SegmentDetectionError

    log = {
        "trajectory_id": "no_hit",
        "frames": [
            _frame(0, 0.0),
            _frame(1, 0.5, collisions=[_collision("FloorBall", "Pillar", 0.0)]),
            _frame(2, 1.0),
        ],
    }
    try:
        detect_segments(log)
    except SegmentDetectionError as e:
        assert "ball impact" in str(e).lower()
        return
    raise AssertionError("expected SegmentDetectionError")


def test_detect_segments_raises_when_no_domino_falls():
    from tools.detect_beamdomino_segments import detect_segments, SegmentDetectionError

    log = {
        "trajectory_id": "no_cascade",
        "frames": [
            _frame(0, 0.0),
            _frame(1, 0.5, collisions=[_collision("FloorBall", "Pillar", 0.5)]),
            _frame(2, 1.0),
            _frame(3, 1.5),
        ],
    }
    try:
        detect_segments(log)
    except SegmentDetectionError as e:
        assert "fallen" in str(e).lower()
        return
    raise AssertionError("expected SegmentDetectionError")
