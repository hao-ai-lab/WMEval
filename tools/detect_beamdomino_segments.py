"""Detect BeamDomino's four canonical causal segment boundaries from a
GroundTruthRecorder per-frame physics log.

The boundaries are:
  rolling : t=0           -> first frame with a nonzero-impulse Ball collision
  impact  : ball-impact   -> first frame with >=1 fallen domino
  topple  : first fall    -> cascade plateau (no new fall for plateau_window_sec)
  cascade : plateau start -> end of log

A "fallen" domino is one whose rotation has tilted past `fallen_tilt_deg`
about its x or z axis (default 30 degrees). A "ball impact" is the first
collision involving a body whose name contains "Ball" with impulse above
`ball_impulse_threshold` (default 0.01) -- this filters out resting/grazing
contacts that report zero or near-zero impulse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


class SegmentDetectionError(RuntimeError):
    pass


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _tilt_deg(rotation: Dict[str, float]) -> float:
    return max(abs(_wrap180(rotation["x"])), abs(_wrap180(rotation["z"])))


def _fallen_count(frame: Dict[str, Any], tilt_deg: float) -> int:
    return sum(
        1
        for b in frame.get("bodies", [])
        if b["name"].startswith("Domino") and _tilt_deg(b["rotation"]) > tilt_deg
    )


def _is_ball_impact(coll: Dict[str, Any], impulse_threshold: float) -> bool:
    a = coll.get("bodyA", "")
    b = coll.get("bodyB", "")
    if "Ball" not in a and "Ball" not in b:
        return False
    return float(coll.get("impulse", 0.0)) > impulse_threshold


def _find_first_frame(frames: List[Dict[str, Any]], predicate) -> int:
    for i, f in enumerate(frames):
        if predicate(f):
            return i
    return -1


def detect_segments(
    log: Dict[str, Any],
    *,
    plateau_window_sec: float = 0.5,
    fallen_tilt_deg: float = 30.0,
    ball_impulse_threshold: float = 0.01,
) -> Dict[str, Any]:
    frames = log["frames"]
    if not frames:
        raise SegmentDetectionError("empty frames list")

    i_hit = _find_first_frame(
        frames,
        lambda f: any(
            _is_ball_impact(c, ball_impulse_threshold)
            for c in (f.get("collisions") or [])
        ),
    )
    if i_hit < 0:
        raise SegmentDetectionError(
            "no ball impact found "
            f"(no collision involving a 'Ball' body with impulse > {ball_impulse_threshold})"
        )

    i_first_fall = _find_first_frame(
        frames, lambda f: _fallen_count(f, fallen_tilt_deg) >= 1
    )
    if i_first_fall < 0:
        raise SegmentDetectionError(
            f"no fallen domino found after impact (tilt threshold {fallen_tilt_deg} deg)"
        )

    plateau_idx = -1
    for i in range(i_first_fall + 1, len(frames)):
        t_i = frames[i]["time"]
        fc_i = _fallen_count(frames[i], fallen_tilt_deg)
        stable = True
        for j in range(i + 1, len(frames)):
            if frames[j]["time"] - t_i > plateau_window_sec:
                break
            if _fallen_count(frames[j], fallen_tilt_deg) > fc_i:
                stable = False
                break
        if stable and frames[-1]["time"] - t_i >= plateau_window_sec - 1e-9:
            plateau_idx = i
            break
    if plateau_idx < 0:
        plateau_idx = len(frames) - 1

    t_hit = frames[i_hit]["time"]
    t_first_fall = frames[i_first_fall]["time"]
    t_plateau = frames[plateau_idx]["time"]
    t_end = frames[-1]["time"]

    events = [
        {"index": 1, "name": "rolling", "start_sec": frames[0]["time"], "end_sec": t_hit,        "trigger": "start"},
        {"index": 2, "name": "impact",  "start_sec": t_hit,             "end_sec": t_first_fall, "trigger": "ball_impulse"},
        {"index": 3, "name": "topple",  "start_sec": t_first_fall,      "end_sec": t_plateau,    "trigger": "first_fallen"},
        {"index": 4, "name": "cascade", "start_sec": t_plateau,         "end_sec": t_end,        "trigger": "cascade_plateau"},
    ]
    return {
        "trajectory_id": log.get("trajectory_id", "unknown"),
        "source": "unity_gt",
        "events": events,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detect BeamDomino canonical segments from a GT log.")
    p.add_argument("--log", required=True, help="Path to unity_gt_log.json")
    p.add_argument("--out", required=True, help="Path to write segments.json")
    p.add_argument("--plateau-window-sec", type=float, default=0.5)
    p.add_argument("--fallen-tilt-deg", type=float, default=30.0,
                   help="Domino is 'fallen' once max(|rot.x|,|rot.z|) exceeds this.")
    p.add_argument("--ball-impulse-threshold", type=float, default=0.01,
                   help="Min collision impulse to count as a ball impact (filters resting contacts).")
    args = p.parse_args(argv)

    with open(args.log) as f:
        log = json.load(f)
    segments = detect_segments(
        log,
        plateau_window_sec=args.plateau_window_sec,
        fallen_tilt_deg=args.fallen_tilt_deg,
        ball_impulse_threshold=args.ball_impulse_threshold,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"wrote {args.out} with {len(segments['events'])} events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
