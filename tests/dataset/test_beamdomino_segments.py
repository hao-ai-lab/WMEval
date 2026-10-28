from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "beamdomino"


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
    from tools.dataset.detect_beamdomino_segments import detect_segments

    with open(FIXTURE_DIR / "tiny_log.json") as f:
        log = json.load(f)
    out = detect_segments(log, plateau_window_sec=0.5)

    with open(FIXTURE_DIR / "tiny_segments.json") as f:
        expected = json.load(f)

    assert out["trajectory_id"] == expected["trajectory_id"]
    assert out["source"] == "unity_gt"
    assert out["trajectory_shape"] == expected["trajectory_shape"]
    assert len(out["events"]) == 4
    for got, want in zip(out["events"], expected["events"]):
        assert got["index"] == want["index"]
        assert got["name"] == want["name"]
        assert got["trigger"] == want["trigger"]
        assert abs(got["start_sec"] - want["start_sec"]) < 1e-6
        assert abs(got["end_sec"] - want["end_sec"]) < 1e-6


def test_detect_segments_raises_when_no_ball_impact():
    from tools.dataset.detect_beamdomino_segments import detect_segments, SegmentDetectionError

    log = {
        "trajectory_id": "no_hit",
        "frames": [
            # Only a startup settling contact, before the default 0.1 s cutoff.
            _frame(0, 0.0),
            _frame(1, 0.02, collisions=[_collision("FloorBall", "Pillar", 0.0)]),
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
    from tools.dataset.detect_beamdomino_segments import detect_segments, SegmentDetectionError

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


def _shape_frame(idx, t, *, roof_tilt=0.0, fallen=0, collisions=()):
    frame = _frame(idx, t, fallen_indices=range(fallen), collisions=collisions)
    frame["bodies"].append(_body("Roof", rotation_x=roof_tilt))
    return frame


def test_detect_segments_glancing_when_roof_never_tips():
    from tools.dataset.detect_beamdomino_segments import detect_segments

    # Ball clips a leg at 0.5 s, the roof stays level, the ball strikes a
    # domino directly at 0.8 s, and the row falls from 1.0 s.
    log = {
        "trajectory_id": "glance",
        "frames": [
            _shape_frame(0, 0.0),
            _shape_frame(1, 0.5, collisions=[_collision("FloorBall", "Pillar", 0.0)]),
            _shape_frame(2, 0.8, collisions=[_collision("FloorBall", "Domino_S00_D00", 0.0)]),
            _shape_frame(3, 1.0, fallen=1),
            _shape_frame(4, 1.2, fallen=3),
            _shape_frame(5, 2.0, fallen=3),
        ],
    }
    out = detect_segments(log)

    assert out["trajectory_shape"] == "glancing"
    assert [(e["name"], e["start_sec"], e["end_sec"]) for e in out["events"]] == [
        ("rolling", 0.0, 0.5),
        ("glance", 0.5, 0.8),
        ("strike", 0.8, 1.0),
        ("cascade", 1.0, 1.2),
    ]


def test_detect_segments_direct_strike_when_ball_hits_a_domino_first():
    from tools.dataset.detect_beamdomino_segments import detect_segments

    # The ball never touches the platform: its first contact is a domino.
    log = {
        "trajectory_id": "far",
        "frames": [
            _shape_frame(0, 0.0),
            _shape_frame(1, 0.6, collisions=[_collision("FloorBall", "Domino_S00_D00", 0.0)]),
            _shape_frame(2, 0.9, fallen=1),
            _shape_frame(3, 1.1, fallen=3),
            _shape_frame(4, 2.0, fallen=3),
        ],
    }
    out = detect_segments(log)

    assert out["trajectory_shape"] == "direct_strike"
    assert [(e["name"], e["start_sec"], e["end_sec"]) for e in out["events"]] == [
        ("rolling", 0.0, 0.6),
        ("strike", 0.6, 0.9),
        ("cascade", 0.9, 1.1),
    ]
