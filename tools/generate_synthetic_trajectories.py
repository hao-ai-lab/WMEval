#!/usr/bin/env python3
"""Generate the WMEval Lite synthetic action-trajectory suite (``test_full``).

Every trajectory is a structured, schema-v1 ``trajectory.yaml`` (the same format
as the curated cases) built from a named pattern and bound to an existing core
case that supplies the scene, conditioning frame, prompt and action target.
Patterns keep the source case's initial state, so the core conditioning frame
stays valid. The suite is deterministic: re-running reproduces byte-identical
files, which ``--verify`` checks against ``synthetic/manifest.json``.

Each trajectory directory gets ``trajectory.yaml``, the compiled ``action.npz``
(timestamps, values, channel names; camera poses for look-at orbits),
``actions.jsonl`` (Project Genie browser program) and ``meta.json``. The
generated files are published to the Hub dataset (split ``test_full``); this
repository keeps the generator and the manifest.

    python tools/generate_synthetic_trajectories.py            # write eval_sets/wmeval_lite_v1/synthetic/
    python tools/generate_synthetic_trajectories.py --verify   # regenerate and compare with the manifest

Unity reference renders for these trajectories are not part of this release:
they are generation-ready, and scoring against simulator ground truth needs a
Unity capture per trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.trajectories import load_trajectory, write_compiled_trajectory  # noqa: E402

SUITE_ROOT = ROOT / "eval_sets/wmeval_lite_v1"
OUT_ROOT = SUITE_ROOT / "synthetic"
MANIFEST = OUT_ROOT / "manifest.json"
GENERATOR_VERSION = 1

Seg = tuple[float, dict[str, float]]  # (duration_sec, channel values)


# --------------------------------------------------------------------------- helpers
def segments(parts: list[Seg], duration: float, hz: int, rest: dict[str, float]) -> list[dict[str, Any]]:
    """Contiguous segments on the control grid; pads with ``rest`` to ``duration``."""
    out, t = [], 0
    total = round(duration * hz)
    for seconds, values in parts:
        steps = max(1, round(seconds * hz))
        steps = min(steps, total - t)
        if steps <= 0:
            break
        out.append({"start_sec": t / hz, "end_sec": (t + steps) / hz, "values": dict(values)})
        t += steps
    if t < total:
        out.append({"start_sec": t / hz, "end_sec": total / hz, "values": dict(rest)})
    return out


def square_wave(channel: str, amp: float, f0: float, f1: float, start: float, end: float, other: dict) -> list[Seg]:
    """Square wave on ``channel`` sweeping linearly from f0 to f1 Hz between start and end."""
    parts: list[Seg] = [(start, {**other, channel: 0.0})] if start > 0 else []
    t, sign = start, 1.0
    while t < end - 1e-9:
        f = f0 + (f1 - f0) * (t - start) / max(end - start, 1e-9)
        half = min(0.5 / f, end - t)
        parts.append((half, {**other, channel: sign * amp}))
        t += half
        sign = -sign
    return parts


def core_case(case_id: str) -> dict[str, Any]:
    return yaml.safe_load((SUITE_ROOT / "cases" / case_id / "case.yaml").read_text(encoding="utf-8"))


def first_frame(case_id: str) -> str:
    ref = core_case(case_id).get("reference") or {}
    rel = ref.get("first_frame") or ref.get("video")
    return f"cases/{case_id}/{rel}"


# --------------------------------------------------------------------------- families
GENIE_YAW = [{"channel": "yaw_rate_deg_s", "threshold": 1.0, "positive_key": "ArrowRight", "negative_key": "ArrowLeft"}]


def camera_yaw_patterns() -> dict[str, list[Seg]]:
    y = "yaw_rate_deg_s"
    return {
        "turn_left_18": [(10, {y: -18})],
        "turn_right_18": [(10, {y: 18})],
        "turn_left_60": [(10, {y: -60})],
        "turn_right_60": [(10, {y: 60})],
        "perturb_left_30_return": [(1, {y: 0}), (1, {y: -30}), (2, {y: 0}), (1, {y: 30})],
        "perturb_right_45_return": [(1, {y: 0}), (1.5, {y: 30}), (2, {y: 0}), (1.5, {y: -30})],
        "sweep_pm45": [(1.5, {y: -30}), (3, {y: 30}), (1.5, {y: -30})],
        "stop_and_go_left": [(2, {y: -36}), (1, {y: 0})] * 3,
        "accelerating_left": [(3, {y: -12}), (3, {y: -36}), (3, {y: -72})],
        "reverse_mid_turn": [(5, {y: -36}), (5, {y: 36})],
    }


CAMERA_YAW_CONTEXTS = [  # (core case supplying frame + prompt)
    "WanderingGarden/scene_360_in", "WanderingGarden/scene_360_out", "BilliardGame/scene_360_in",
    "MirrorRollingBall/prompt_first", "MirrorRollingBall/prompt_third",
    "ChessBoard/prompt_first", "ChessBoard/prompt_third",
    "RacingCar/prompt_first", "RacingCar/prompt_third",
]


def orbit_patterns() -> dict[str, list[Seg]]:
    r = "orbit_phase_rate_deg_s"
    return {
        "cw_full": [(10, {r: 36})],
        "ccw_full": [(10, {r: -36})],
        "cw_half_hold": [(5, {r: 36})],
        "ccw_half_hold": [(5, {r: -36})],
        "cw_one_and_half": [(10, {r: 54})],
        "ccw_one_and_half": [(10, {r: -54})],
        "cw_ccw_return": [(5, {r: 36}), (5, {r: -36})],
        "cw_stop_and_go": [(2, {r: 45}), (1, {r: 0})] * 3,
        "cw_quarter_slow": [(10, {r: 9})],
        "ccw_quarter_slow": [(10, {r: -9})],
        "cw_accelerating": [(4, {r: 18}), (3, {r: 36}), (3, {r: 54})],
        "ccw_accelerating": [(4, {r: -18}), (3, {r: -36}), (3, {r: -54})],
    }


def car_patterns(k: float) -> dict[str, list[Seg]]:
    def s(steer, throttle=0.8, brake=0):
        return {"steer": steer, "throttle": throttle, "brake": brake}
    return {
        "lane_change_left": [(1, s(0, 0)), (2, s(0)), (1, s(-k)), (1, s(k)), (5, s(0))],
        "lane_change_right": [(1, s(0, 0)), (2, s(0)), (1, s(k)), (1, s(-k)), (5, s(0))],
        "double_lane_change": [(1, s(0, 0)), (1.5, s(0)), (1, s(-k)), (1, s(k)), (1.5, s(0)), (1, s(k)), (1, s(-k)), (2, s(0))],
        "slalom_three": [(1, s(0, 0)), (1, s(0))] + [(1, s(-k)), (1, s(k))] * 3 + [(2, s(0))],
        "accelerate_brake": [(1, s(0, 0)), (4, s(0, 1.0)), (2, s(0, 0, 1)), (3, s(0, 0))],
        "cruise_brake_turn": [(1, s(0, 0)), (3, s(0)), (1.5, s(0, 0, 1)), (1.5, s(k * 0.6)), (3, s(0))],
        "s_curve": [(1, s(0, 0)), (1, s(0)), (3, s(k * 0.5)), (3, s(-k * 0.5)), (2, s(0))],
        "launch_coast": [(1, s(0, 0)), (2, s(0, 1.0)), (7, s(0, 0))],
    }


def cue_patterns(m: float) -> dict[str, list[Seg]]:
    def b(f, l=0.0):
        return {"forward": f, "lateral": l}
    z = b(0)
    return {
        "forward_tap": [(1, z), (0.5, b(m)), (8.5, z)],
        "forward_hold": [(1, z), (2, b(m)), (7, z)],
        "diagonal_left": [(1, z), (1, b(m, -m)), (8, z)],
        "diagonal_right": [(1, z), (1, b(m, m)), (8, z)],
        "zigzag": [(1, z)] + [(0.5, b(m, -m)), (0.5, b(m, m))] * 3 + [(6, z)],
        "square": [(1, z), (0.6, b(m)), (0.6, b(0, m)), (0.6, b(-m)), (0.6, b(0, -m)), (6.6, z)],
        "back_and_forth": [(1, z), (0.8, b(m)), (1, z), (0.8, b(-m)), (6.4, z)],
    }


def ball_patterns() -> dict[str, list[Seg]]:
    def v(h, w):
        return {"horizontal": h, "vertical": w}
    z = v(0, 0)
    d = 1.0 / math.sqrt(2.0)
    compass = {"north": v(0, 1), "northeast": v(d, d), "east": v(1, 0), "southeast": v(d, -d),
               "south": v(0, -1), "southwest": v(-d, -d), "west": v(-1, 0), "northwest": v(-d, d)}
    pats: dict[str, list[Seg]] = {f"straight_{name}": [(0.5, z), (2.0, val)] for name, val in compass.items()}
    pats["square_loop"] = [(0.5, z), (1, v(0, 1)), (1, v(1, 0)), (1, v(0, -1)), (1, v(-1, 0))]
    pats["zigzag_north"] = [(0.5, z)] + [(0.5, v(d, d)), (0.5, v(-d, d))] * 3
    pats["figure_eight"] = [(0.5, z), (0.6, v(d, d)), (0.6, v(d, -d)), (0.6, v(-d, -d)), (0.6, v(-d, d))] * 1 + \
                           [(0.6, v(-d, d)), (0.6, v(-d, -d)), (0.6, v(d, -d)), (0.6, v(d, d))]
    pats["stop_and_go_east"] = [(0.5, z)] + [(0.6, v(1, 0)), (0.6, z)] * 3
    return pats


def beam_patterns() -> dict[str, list[Seg]]:
    def f(zf, xf=0.0):
        return {"z_force": zf, "x_force": xf}
    z = f(0)
    pats: dict[str, list[Seg]] = {}
    for delay in (1, 2, 3):
        for push in (3, 5):
            pats[f"straight_delay{delay}_push{push}"] = [(delay, z), (push, f(1))]
    for xf in (-0.5, -0.25, 0.25, 0.5):
        name = f"angled_{'left' if xf < 0 else 'right'}_{abs(xf):.2f}".replace(".", "p")
        pats[name] = [(2, z), (5, f(1, xf))]
    pats["late_strike"] = [(6, z), (4, f(1))]
    pats["double_push"] = [(2, z), (2, f(1)), (1, z), (2, f(1))]
    pats["double_push_angled"] = [(2, z), (2, f(1, 0.3)), (1, z), (2, f(1, -0.3))]
    pats["reverse_then_forward"] = [(1, z), (1, f(-1)), (1, z), (5, f(1))]
    pats["strafe_then_forward_left"] = [(1, z), (1.5, f(0, -1)), (5, f(1))]
    pats["strafe_then_forward_right"] = [(1, z), (1.5, f(0, 1)), (5, f(1))]
    return pats


def liquid_patterns() -> dict[str, list[Seg]]:
    def m(fw=0.0, rt=0.0, up=0.0):
        return {"forward": fw, "right": rt, "up": up}
    z = m()
    pats: dict[str, list[Seg]] = {
        "step_right_2s": [(2, z), (2, m(rt=1))],
        "step_right_4s": [(2, z), (4, m(rt=1))],
        "step_right_8s": [(2, z), (8, m(rt=1))],
        "step_left_4s": [(2, z), (4, m(rt=-1))],
        "step_forward_4s": [(2, z), (4, m(fw=1))],
        "step_diagonal_4s": [(2, z), (4, m(fw=1, rt=1))],
        "impulse_right_3s": [(3, z), (0.25, m(rt=1))],
        "impulse_right_9s": [(9, z), (0.25, m(rt=1))],
        "impulse_right_15s": [(15, z), (0.25, m(rt=1))],
        "impulse_forward_9s": [(9, z), (0.25, m(fw=1))],
        "impulse_up_6s": [(6, z), (0.25, m(up=1))],
        "up_bounces": [(2, z)] + [(0.25, m(up=1)), (1.75, z)] * 4,
        "forward_back_shuttle": [(2, z)] + [(1.5, m(fw=1)), (1.5, m(fw=-1))] * 3,
    }
    pats["chirp_right_0p3_1p5hz"] = square_wave("right", 1.0, 0.3, 1.5, 2, 18, z)
    pats["chirp_right_0p5_3hz"] = square_wave("right", 1.0, 0.5, 3.0, 2, 18, z)
    pats["chirp_forward_0p3_2hz"] = square_wave("forward", 1.0, 0.3, 2.0, 2, 18, z)
    for f in (0.5, 1.0, 2.0):
        pats[f"square_right_{str(f).replace('.', 'p')}hz"] = square_wave("right", 1.0, f, f, 2, 14, z)
    pats["square_forward_1hz"] = square_wave("forward", 1.0, 1.0, 1.0, 2, 14, z)
    return pats


def character_patterns() -> dict[str, list[Seg]]:
    def c(x=0.0, y=0.0, jump=0, sprint=0):
        return {"move_x": x, "move_y": y, "jump": jump, "sprint": sprint}
    jump_step = 1 / 30
    return {
        "walk_forward": [(6, c(y=1))],
        "walk_backward": [(6, c(y=-1))],
        "strafe_left": [(6, c(x=-1))],
        "strafe_right": [(6, c(x=1))],
        "walk_turn_left": [(2, c(y=1)), (2, c(x=-0.8, y=0.6)), (2, c(y=1))],
        "walk_turn_right": [(2, c(y=1)), (2, c(x=0.8, y=0.6)), (2, c(y=1))],
        "sprint_forward": [(1, c(y=1)), (5, c(y=1, sprint=1))],
        "zigzag": [(1, c(x=0.7, y=0.7)), (1, c(x=-0.7, y=0.7))] * 3,
        "jumps_while_walking": [(1, c(y=1)), (jump_step, c(y=1, jump=1)), (1.5, c(y=1)),
                                (jump_step, c(y=1, jump=1)), (1.5, c(y=1)), (jump_step, c(y=1, jump=1)), (2, c(y=1))],
        "circle": [(0.75, c(x=dx, y=dy)) for dx, dy in
                   [(0, 1), (0.7, 0.7), (1, 0), (0.7, -0.7), (0, -1), (-0.7, -0.7), (-1, 0), (-0.7, 0.7)]],
        "stop_and_go": [(1, c(y=1)), (1, c())] * 3,
        "sprint_then_walk": [(3, c(y=1, sprint=1)), (3, c(y=1))],
    }


# (family, target, core case(s), duration, control_hz, channels, rest, genie key rules, patterns)
def families() -> list[dict[str, Any]]:
    cont = lambda unit="normalized": {"kind": "continuous", "min": -1.0, "max": 1.0, "unit": unit}  # noqa: E731
    boolean = {"kind": "discrete", "min": 0, "max": 1, "unit": "boolean"}
    fams = [
        {"family": "camera_yaw", "target": "camera", "cases": CAMERA_YAW_CONTEXTS, "duration": 10.0, "hz": 20,
         "channels": {"yaw_rate_deg_s": {"kind": "continuous", "min": -360.0, "max": 360.0, "unit": "deg_per_sec"}},
         "keys": GENIE_YAW, "patterns": camera_yaw_patterns(), "dimensions": ["environment_object_consistency", "prompt_scene_consistency"]},
        {"family": "camera_orbit", "target": "camera", "cases": ["SunRisesAndSets/lighting_orbit_v2"], "duration": 10.0,
         "hz": 60, "orbit": True, "patterns": orbit_patterns(), "dimensions": ["lighting_shading"]},
        {"family": "ego_vehicle", "target": "ego_vehicle", "cases": ["RacingCar/kinematics_third"], "duration": 10.0, "hz": 30,
         "channels": {"steer": cont(), "throttle": cont(), "brake": boolean},
         "keys": [{"channel": "steer", "threshold": 0.25, "positive_key": "KeyD", "negative_key": "KeyA"},
                  {"channel": "throttle", "threshold": 0.25, "positive_key": "KeyW", "negative_key": "KeyS"},
                  {"channel": "brake", "threshold": 0.5, "positive_key": "KeyS"}],
         "patterns": {f"{name}_steer{k:.1f}".replace(".", "p"): p
                      for k in (0.4, 0.7, 1.0) for name, p in car_patterns(k).items()},
         "dimensions": ["kinematics_mechanics", "environment_object_consistency"]},
        {"family": "cue_ball", "target": "cue_ball", "cases": ["BilliardGame/kinematics_first"], "duration": 10.0, "hz": 30,
         "channels": {"forward": cont("normalized_ball_impulse"), "lateral": cont("normalized_ball_impulse")},
         "keys": [{"channel": "forward", "threshold": 0.25, "positive_key": "KeyW", "negative_key": "KeyS"},
                  {"channel": "lateral", "threshold": 0.15, "positive_key": "KeyD", "negative_key": "KeyA"}],
         "patterns": {f"{name}_mag{m:.2f}".replace(".", "p"): p
                      for m in (0.4, 0.65, 0.9) for name, p in cue_patterns(m).items()},
         "dimensions": ["kinematics_mechanics"]},
        {"family": "ball_force", "target": "ball", "cases": ["BallRoller/ctrl", "MirrorRollingBall/lpips_collision"],
         "duration": 6.0, "hz": 20,
         "channels": {"horizontal": cont("normalized_force_input"), "vertical": cont("normalized_force_input")},
         "keys": [{"channel": "horizontal", "threshold": 0.25, "positive_key": "KeyD", "negative_key": "KeyA"},
                  {"channel": "vertical", "threshold": 0.25, "positive_key": "KeyW", "negative_key": "KeyS"}],
         "patterns": ball_patterns(), "dimensions": ["action_controllability", "lpips"]},
        {"family": "beam_domino", "target": "rolling_ball", "cases": ["BeamDomino/causal"], "duration": 12.2, "hz": 10,
         "channels": {"z_force": cont(), "x_force": cont()},
         "keys": [{"channel": "z_force", "threshold": 0.25, "positive_key": "KeyW", "negative_key": "KeyS"},
                  {"channel": "x_force", "threshold": 0.25, "positive_key": "KeyD", "negative_key": "KeyA"}],
         "patterns": beam_patterns(), "dimensions": ["physical_causality", "action_controllability", "lpips"]},
        {"family": "liquid_container", "target": "liquid_container", "cases": ["LiquidContainer/fluid_rest"],
         "duration": 20.0, "hz": 12, "channels": {"forward": cont(), "right": cont(), "up": cont()},
         "keys": [{"channel": "forward", "threshold": 0.5, "positive_key": "KeyW", "negative_key": "KeyS"},
                  {"channel": "right", "threshold": 0.5, "positive_key": "KeyD", "negative_key": "KeyA"},
                  {"channel": "up", "threshold": 0.5, "positive_key": "Space"}],
         "patterns": liquid_patterns(), "dimensions": ["fluid_mechanics", "lpips"]},
        {"family": "character", "target": "character", "cases": ["ChessBoard/walk_third"], "duration": 6.0, "hz": 30,
         "channels": {"move_x": cont(), "move_y": cont(), "jump": boolean, "sprint": boolean},
         "keys": [{"channel": "move_x", "threshold": 0.4, "positive_key": "KeyD", "negative_key": "KeyA"},
                  {"channel": "move_y", "threshold": 0.4, "positive_key": "KeyW", "negative_key": "KeyS"},
                  {"channel": "jump", "threshold": 0.5, "positive_key": "Space"},
                  {"channel": "sprint", "threshold": 0.5, "positive_key": "Shift"}],
         "patterns": character_patterns(), "dimensions": ["environment_object_consistency"]},
    ]
    return fams


# --------------------------------------------------------------------------- build
def trajectory_doc(fam: dict[str, Any], case_id: str, pattern: str, parts: list[Seg]) -> dict[str, Any]:
    scene, variant = case_id.split("/")
    tid = f"synthetic.{fam['family']}.{scene.lower()}_{variant}.{pattern}_v1"
    description = (f"Synthetic {fam['family'].replace('_', ' ')} pattern '{pattern}' on {case_id}, "
                   f"generated by tools/generate_synthetic_trajectories.py (v{GENERATOR_VERSION}); "
                   "starts from the core case's initial state.")
    if fam.get("orbit"):
        base = load_trajectory(SUITE_ROOT / "cases" / case_id / "reference/artifacts/trajectory.yaml")
        rate = "orbit_phase_rate_deg_s"
        doc = {
            "schema_version": 1, "trajectory_id": tid, "description": description,
            "kind": "camera_look_at_orbit", "target": "camera", "coordinates": base["coordinates"],
            "duration_sec": fam["duration"], "control_hz": fam["hz"], "geometry": base["geometry"],
            "channels": {rate: base["channels"][rate]},
            "segments": segments(parts, fam["duration"], fam["hz"], {rate: 0.0}),
            "environment": base.get("environment"),
        }
        for key in ("adapters", "paper_alignment"):
            if key in base:
                doc[key] = base[key]
        doc["paper_alignment"] = {"status": "synthetic", "note": "Not in the paper; procedurally generated."}
        return doc
    rest = {name: 0.0 if spec["kind"] == "continuous" else 0 for name, spec in fam["channels"].items()}
    clean = [(d, {k: (int(v) if fam["channels"][k]["kind"] == "discrete" else float(v)) for k, v in vals.items()})
             for d, vals in parts]
    return {
        "schema_version": 1, "trajectory_id": tid, "description": description,
        "duration_sec": fam["duration"], "control_hz": fam["hz"], "channels": fam["channels"],
        "segments": segments(clean, fam["duration"], fam["hz"], rest),
        "paper_alignment": {"status": "synthetic", "matched": [], "missing": "Not in the paper; procedurally generated.",
                            "source": f"cases/{case_id} (initial state)"},
        "adapters": {"unity": {"status": "pending_render", "channels": list(fam["channels"])},
                     "project_genie": {"status": "nominal", "key_rules": fam["keys"]}},
    }


def generate(out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fam in families():
        for case_id in fam["cases"]:
            case = core_case(case_id)
            scene, variant = case_id.split("/")
            for pattern, parts in fam["patterns"].items():
                doc = trajectory_doc(fam, case_id, pattern, parts)
                rel = f"{fam['family']}/{scene}__{variant}/{pattern}"
                out = out_root / rel
                out.mkdir(parents=True, exist_ok=True)
                traj = out / "trajectory.yaml"
                traj.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
                # Compile next to the source so the recorded source path is location-independent.
                write_compiled_trajectory(traj, out)
                (out / "trajectory.action.npz").replace(out / "action.npz")
                if (out / "trajectory.actions.jsonl").is_file():
                    (out / "trajectory.actions.jsonl").replace(out / "actions.jsonl")
                if (out / "trajectory.poses.npy").is_file():  # camera poses of look-at orbits
                    (out / "trajectory.poses.npy").replace(out / "poses.npy")
                for extra in ("trajectory.action.npy", "trajectory.actions.txt", "trajectory.manifest.json"):
                    (out / extra).unlink(missing_ok=True)
                meta = {
                    "schema_version": 1, "synthetic_id": f"synthetic/{rel}", "trajectory_id": doc["trajectory_id"],
                    "family": fam["family"], "pattern": pattern, "action_target": fam["target"],
                    "scene": scene, "conditioning_case": case_id, "perspective": case.get("perspective"),
                    "prompt": case.get("prompt"), "first_frame": first_frame(case_id),
                    "duration_sec": fam["duration"], "control_hz": fam["hz"], "dimensions": fam["dimensions"],
                    "reference_status": "pending_unity_render", "generator_version": GENERATOR_VERSION,
                }
                (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
                rows.append({**meta, "path": f"synthetic/{rel}", "files": files})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="regenerate in a temp dir and compare with the manifest")
    args = parser.parse_args(argv)
    if args.verify:
        with tempfile.TemporaryDirectory() as tmp:
            rows = generate(Path(tmp) / "synthetic")
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["trajectories"]
        got = {r["synthetic_id"]: r["files"] for r in rows}
        want = {r["synthetic_id"]: r["files"] for r in expected}
        if got != want:
            diff = sorted(set(got) ^ set(want)) or [k for k in got if got[k] != want.get(k)]
            print(f"MISMATCH: {len(diff)} trajectory(ies) differ, e.g. {diff[:3]}")
            return 1
        print(f"verified {len(rows)} synthetic trajectories against {MANIFEST.relative_to(ROOT)}")
        return 0
    if OUT_ROOT.exists():
        for child in OUT_ROOT.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
    rows = generate(OUT_ROOT)
    by_family: dict[str, int] = {}
    for r in rows:
        by_family[r["family"]] = by_family.get(r["family"], 0) + 1
    MANIFEST.write_text(json.dumps({"schema_version": 1, "generator_version": GENERATOR_VERSION,
                                    "count": len(rows), "by_family": by_family, "trajectories": rows},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(rows), "by_family": by_family}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
