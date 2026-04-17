"""Detect BeamDomino's four canonical causal segment boundaries from a
GroundTruthRecorder per-frame physics log.

The boundaries are:
  rolling : t=0  -> first frame with stack_hit_flag == 1
  impact  : stack_hit_flag -> first frame with fallen_count >= 1
  topple  : first_fallen  -> cascade plateau (no new fall for plateau_window_sec)
  cascade : plateau_start -> end of log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


class SegmentDetectionError(RuntimeError):
    pass


def _find_first_frame(frames: List[Dict[str, Any]], predicate) -> int:
    for i, f in enumerate(frames):
        if predicate(f):
            return i
    return -1


def detect_segments(log: Dict[str, Any], plateau_window_sec: float = 0.5) -> Dict[str, Any]:
    frames = log["frames"]
    if not frames:
        raise SegmentDetectionError("empty frames list")

    i_hit = _find_first_frame(frames, lambda f: int(f.get("stack_hit_flag", 0)) >= 1)
    if i_hit < 0:
        raise SegmentDetectionError("stack_hit_flag never became 1")

    i_first_fall = _find_first_frame(frames, lambda f: int(f.get("fallen_count", 0)) >= 1)
    if i_first_fall < 0:
        raise SegmentDetectionError("no fallen_count >= 1 after impact")

    # Find cascade plateau: the first frame where fallen_count hasn't increased
    # for at least plateau_window_sec.
    plateau_idx = -1
    for i in range(i_first_fall + 1, len(frames)):
        t_i = frames[i]["t"]
        fc_i = frames[i].get("fallen_count", 0)
        stable = True
        for j in range(i + 1, len(frames)):
            if frames[j]["t"] - t_i > plateau_window_sec:
                break
            if frames[j].get("fallen_count", 0) > fc_i:
                stable = False
                break
        else:
            # reached end without seeing growth -> plateau
            stable = True
        # Need at least plateau_window_sec of future log to confirm; if last
        # frame is within that window, treat it as plateau at last frame.
        if stable and frames[-1]["t"] - t_i >= plateau_window_sec - 1e-9:
            plateau_idx = i
            break
    if plateau_idx < 0:
        # cascade didn't plateau inside the log; treat the final frame as plateau
        plateau_idx = len(frames) - 1

    t_hit = frames[i_hit]["t"]
    t_first_fall = frames[i_first_fall]["t"]
    t_plateau = frames[plateau_idx]["t"]
    t_end = frames[-1]["t"]

    events = [
        {"index": 1, "name": "rolling", "start_sec": frames[0]["t"], "end_sec": t_hit,       "trigger": "start"},
        {"index": 2, "name": "impact",  "start_sec": t_hit,         "end_sec": t_first_fall, "trigger": "stack_hit_flag"},
        {"index": 3, "name": "topple",  "start_sec": t_first_fall,  "end_sec": t_plateau,    "trigger": "first_fallen_flag"},
        {"index": 4, "name": "cascade", "start_sec": t_plateau,     "end_sec": t_end,        "trigger": "cascade_plateau"},
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
    args = p.parse_args(argv)

    with open(args.log) as f:
        log = json.load(f)
    segments = detect_segments(log, plateau_window_sec=args.plateau_window_sec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"wrote {args.out} with {len(segments['events'])} events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
