"""Detect BeamDomino causal segment boundaries from a GroundTruthRecorder
per-frame physics log.

Three trajectory shapes are recognised and emit different segment chains so
each QA only asks about events that actually happened:

  canonical    rolling -> impact -> topple -> cascade   (4 segments)
                  Roof tilts past `platform_tilt_threshold_deg`; ball hit a
                  Pillar before the first domino fell.

  glancing     rolling -> glance -> strike -> cascade   (4 segments)
                  Ball touched a Pillar/Roof first, but the Roof never
                  passed the tilt threshold; ball then hit a domino
                  directly. Platform "survives".

  direct_strike rolling -> strike -> cascade            (3 segments)
                  Ball's first non-ground contact is with a Domino — it
                  bypassed the platform entirely. No Roof tilt observed.

Boundaries:

  rolling : motion-onset (FloorBall.|v| > thresh) -> ball's first relevant contact
  impact  : ball-Pillar contact   -> Roof tilt > platform_tilt_threshold_deg
  glance  : ball-Pillar contact   -> ball-Domino contact
  topple  : Roof tilt threshold   -> first fallen domino
  strike  : ball-Domino contact   -> first fallen domino
  cascade : first fallen domino   -> chain plateau

A "fallen" domino has tilted past `fallen_tilt_deg` about x or z (default
30 deg). A "ball impact" is the first FloorBall non-self/non-Ground
collision after `ball_impact_min_time_sec` with impulse >=
`ball_impulse_threshold`. Motion onset requires the GT log to expose
``bodies[].velocity``; if missing the rolling segment starts at t=0 so old
fixtures still parse.
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
    return float(coll.get("impulse", 0.0)) >= impulse_threshold


def _floorball_contact_target(coll: Dict[str, Any]) -> str | None:
    """If `coll` is a FloorBall<->X contact (X != Ground/FloorBall/RoofBall),
    return X's body name. Otherwise None.

    Used to classify shape: which kind of object did the ball hit *first*
    after motion onset? Pillar -> canonical/glancing; Domino -> direct_strike.
    """
    a = coll.get("bodyA", "")
    b = coll.get("bodyB", "")
    if "FloorBall" not in a and "FloorBall" not in b:
        return None
    other = b if "FloorBall" in a else a
    if other in {"Ground", ""} or other.startswith(("FloorBall", "RoofBall")):
        return None
    return other


def _find_first_frame(frames: List[Dict[str, Any]], predicate) -> int:
    for i, f in enumerate(frames):
        if predicate(f):
            return i
    return -1


def _floor_ball_speed(frame: Dict[str, Any]) -> float | None:
    """Return |FloorBall.velocity| if the field is present, else ``None``.

    Older fixtures omit the velocity field entirely; in that case the caller
    must fall back to a legacy boundary rather than treating "no velocity"
    as "speed zero" (which would put motion-onset at t=0 forever).
    """
    fb = next((b for b in frame.get("bodies", []) if b.get("name") == "FloorBall"), None)
    if fb is None or "velocity" not in fb:
        return None
    v = fb["velocity"]
    return (float(v["x"]) ** 2 + float(v["y"]) ** 2 + float(v["z"]) ** 2) ** 0.5


def _roof_tilt_deg(frame: Dict[str, Any]) -> float | None:
    """Return Roof body's max-axis tilt in degrees, or ``None`` if no Roof."""
    rf = next((b for b in frame.get("bodies", []) if b.get("name") == "Roof"), None)
    if rf is None:
        return None
    return _tilt_deg(rf["rotation"])


def detect_segments(
    log: Dict[str, Any],
    *,
    plateau_window_sec: float = 0.5,
    fallen_tilt_deg: float = 30.0,
    ball_impulse_threshold: float = 0.01,
    ball_impact_min_time_sec: float = 0.0,
    motion_onset_velocity_mps: float = 0.5,
    platform_tilt_threshold_deg: float = 5.0,
) -> Dict[str, Any]:
    frames = log["frames"]
    if not frames:
        raise SegmentDetectionError("empty frames list")

    # Pre-v2 capture logs used a compact schema (``t``,
    # ``stack_hit_flag``, ``fallen_count``, ``cascade_count``) instead of
    # per-body states and collision records. Keep those committed fixtures
    # and historical captures reproducible rather than trying to interpret
    # them as the richer GroundTruthRecorder schema below.
    if "time" not in frames[0] and "t" in frames[0]:
        i_hit = _find_first_frame(
            frames, lambda f: bool(f.get("stack_hit_flag", 0))
        )
        if i_hit < 0:
            raise SegmentDetectionError("no ball impact found (stack_hit_flag never set)")
        i_first_fall = _find_first_frame(
            frames, lambda f: int(f.get("fallen_count", 0)) >= 1
        )
        if i_first_fall < 0:
            raise SegmentDetectionError("no fallen domino found (fallen_count stayed zero)")

        plateau_idx = len(frames) - 1
        for i in range(i_first_fall, len(frames)):
            value = float(frames[i].get("cascade_count", 0.0))
            if all(
                float(later.get("cascade_count", 0.0)) == value
                for later in frames[i:]
            ):
                plateau_idx = i
                break

        t_start = float(frames[0]["t"])
        t_hit = float(frames[i_hit]["t"])
        t_first_fall = float(frames[i_first_fall]["t"])
        t_plateau = float(frames[plateau_idx]["t"])
        t_end = float(frames[-1]["t"])
        return {
            "trajectory_id": log.get("trajectory_id", "unknown"),
            "source": "unity_gt",
            "trajectory_shape": "canonical",
            "events": [
                {"index": 1, "name": "rolling", "start_sec": t_start,
                 "end_sec": t_hit, "trigger": "start"},
                {"index": 2, "name": "impact", "start_sec": t_hit,
                 "end_sec": t_first_fall, "trigger": "stack_hit_flag"},
                {"index": 3, "name": "topple", "start_sec": t_first_fall,
                 "end_sec": t_plateau, "trigger": "first_fallen_flag"},
                {"index": 4, "name": "cascade", "start_sec": t_plateau,
                 "end_sec": t_end, "trigger": "cascade_plateau"},
            ],
        }

    # ── ball impact ───────────────────────────────────────────────────────
    # Skip the startup window before searching for ball impact. Unity reports
    # initial RoofBall<->Roof / FloorBall<->Ground "settling" contacts at t≈0.02
    # with zero impulse; on some trajectories (e.g. glancing/far hits) the real
    # FloorBall<->Pillar contact also reports impulse=0.0, so callers can't
    # always rely on the impulse threshold alone — they need a time floor too.
    def _ball_hit(f: Dict[str, Any]) -> bool:
        if f["time"] < ball_impact_min_time_sec:
            return False
        return any(
            _is_ball_impact(c, ball_impulse_threshold)
            for c in (f.get("collisions") or [])
        )

    i_hit = _find_first_frame(frames, _ball_hit)
    if i_hit < 0:
        raise SegmentDetectionError(
            "no ball impact found "
            f"(no collision involving a 'Ball' body with impulse > {ball_impulse_threshold} "
            f"after t={ball_impact_min_time_sec}s)"
        )

    # ── first fallen domino ───────────────────────────────────────────────
    i_first_fall = _find_first_frame(
        frames, lambda f: _fallen_count(f, fallen_tilt_deg) >= 1
    )
    if i_first_fall < 0:
        raise SegmentDetectionError(
            f"no fallen domino found after impact (tilt threshold {fallen_tilt_deg} deg)"
        )

    # ── plateau (chain reaction stops propagating) ────────────────────────
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

    # ── motion onset (FloorBall starts moving) ────────────────────────────
    # Falls back to t=0 if the log doesn't expose velocities, which keeps
    # legacy fixtures (no velocity field) parsing the same way they used to.
    has_velocity = _floor_ball_speed(frames[0]) is not None
    if has_velocity:
        i_motion = _find_first_frame(
            frames,
            lambda f: (
                f["time"] >= ball_impact_min_time_sec
                and (s := _floor_ball_speed(f)) is not None
                and s >= motion_onset_velocity_mps
            ),
        )
        if i_motion < 0 or i_motion >= i_hit:
            # Motion never crossed the threshold before impact — clamp to t=0
            # rather than fail; this scene's ball might have very low velocity.
            i_motion = 0
    else:
        i_motion = 0

    # ── shape classification ──────────────────────────────────────────────
    # What did the ball hit *first* (after the startup window)?
    #   Pillar/Roof contact → canonical or glancing (depending on whether
    #     the Roof later tips past the threshold)
    #   Domino contact      → direct_strike (ball bypassed the platform)
    first_target = None
    i_first_pillar = -1
    i_first_domino = -1
    for i in range(i_hit, len(frames)):
        for c in frames[i].get("collisions") or []:
            tgt = _floorball_contact_target(c)
            if tgt is None:
                continue
            if first_target is None:
                first_target = tgt
            if i_first_pillar < 0 and (tgt == "Pillar" or tgt == "Roof"):
                i_first_pillar = i
            if i_first_domino < 0 and tgt.startswith("Domino"):
                i_first_domino = i
        if first_target is not None and i_first_domino >= 0:
            break

    # Roof tilt observed anywhere?
    has_roof = _roof_tilt_deg(frames[0]) is not None
    max_roof_tilt = (
        max((_roof_tilt_deg(f) or 0.0) for f in frames) if has_roof else 0.0
    )

    has_pillar_contact = i_first_pillar >= 0
    if not has_roof and not has_pillar_contact:
        # Legacy/minimal fixtures with no Pillar contact and no Roof body
        # cannot be meaningfully classified — fall back to canonical so old
        # tests parse with the synthesised topple midpoint.
        shape = "canonical"
    elif first_target is not None and first_target.startswith("Domino"):
        shape = "direct_strike"
    elif max_roof_tilt >= platform_tilt_threshold_deg:
        shape = "canonical"
    else:
        shape = "glancing"

    t_motion = frames[i_motion]["time"]
    t_hit = frames[i_hit]["time"]
    t_first_fall = frames[i_first_fall]["time"]
    t_plateau = frames[plateau_idx]["time"]

    if shape == "canonical":
        # Locate roof-tilt onset for the impact->topple boundary.
        i_tilt = _find_first_frame(
            frames,
            lambda f, _i_hit=i_hit: (
                frames.index(f) >= _i_hit
                and (t := _roof_tilt_deg(f)) is not None
                and t >= platform_tilt_threshold_deg
            ),
        )
        if i_tilt < 0 or i_tilt >= i_first_fall:
            # Tilt threshold never crossed before first fallen domino —
            # synthesise a midpoint so the topple segment has nonzero span.
            t_tilt = (t_hit + t_first_fall) / 2.0
        else:
            t_tilt = frames[i_tilt]["time"]
        events = [
            {"index": 1, "name": "rolling", "start_sec": t_motion,     "end_sec": t_hit,        "trigger": "motion_onset"},
            {"index": 2, "name": "impact",  "start_sec": t_hit,        "end_sec": t_tilt,       "trigger": "ball_impulse"},
            {"index": 3, "name": "topple",  "start_sec": t_tilt,       "end_sec": t_first_fall, "trigger": "platform_tilt"},
            {"index": 4, "name": "cascade", "start_sec": t_first_fall, "end_sec": t_plateau,    "trigger": "first_fallen"},
        ]

    elif shape == "glancing":
        # Need a separate ball-domino contact frame to bound the strike segment.
        # If we never found one, fall back to first_fall as the strike anchor —
        # the cascade clearly started somehow, treat first-fall time as the
        # latest possible strike boundary.
        t_strike_start = (
            frames[i_first_domino]["time"] if i_first_domino >= 0 else t_first_fall
        )
        # Glance segment must cover at least one frame; if the glance and
        # strike contacts coincide (ball bounced through everything in one
        # tick) clamp to ensure ordering.
        if t_strike_start <= t_hit:
            t_strike_start = (t_hit + t_first_fall) / 2.0
        events = [
            {"index": 1, "name": "rolling", "start_sec": t_motion,        "end_sec": t_hit,           "trigger": "motion_onset"},
            {"index": 2, "name": "glance",  "start_sec": t_hit,           "end_sec": t_strike_start,  "trigger": "ball_pillar"},
            {"index": 3, "name": "strike",  "start_sec": t_strike_start,  "end_sec": t_first_fall,    "trigger": "ball_domino"},
            {"index": 4, "name": "cascade", "start_sec": t_first_fall,    "end_sec": t_plateau,       "trigger": "first_fallen"},
        ]

    else:  # direct_strike — 3 segments, no platform-encounter middle phase
        events = [
            {"index": 1, "name": "rolling", "start_sec": t_motion,     "end_sec": t_hit,        "trigger": "motion_onset"},
            {"index": 2, "name": "strike",  "start_sec": t_hit,        "end_sec": t_first_fall, "trigger": "ball_domino"},
            {"index": 3, "name": "cascade", "start_sec": t_first_fall, "end_sec": t_plateau,    "trigger": "first_fallen"},
        ]

    return {
        "trajectory_id": log.get("trajectory_id", "unknown"),
        "source": "unity_gt",
        "trajectory_shape": shape,
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
    p.add_argument("--ball-impact-min-time-sec", type=float, default=0.0,
                   help="Skip ball collisions before this sim-time (filters t=0 settling contacts).")
    p.add_argument("--motion-onset-velocity-mps", type=float, default=0.5,
                   help="FloorBall speed (m/s) at which the rolling segment begins. "
                        "Falls back to t=0 if the log has no velocity field.")
    p.add_argument("--platform-tilt-threshold-deg", type=float, default=5.0,
                   help="Roof body tilt (deg) at which the impact->topple boundary fires. "
                        "Falls back to ball-impact frame if there's no Roof body.")
    args = p.parse_args(argv)

    with open(args.log) as f:
        log = json.load(f)
    segments = detect_segments(
        log,
        plateau_window_sec=args.plateau_window_sec,
        fallen_tilt_deg=args.fallen_tilt_deg,
        ball_impulse_threshold=args.ball_impulse_threshold,
        ball_impact_min_time_sec=args.ball_impact_min_time_sec,
        motion_onset_velocity_mps=args.motion_onset_velocity_mps,
        platform_tilt_threshold_deg=args.platform_tilt_threshold_deg,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"wrote {args.out} with {len(segments['events'])} events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
