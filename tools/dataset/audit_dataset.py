#!/usr/bin/env python3
"""Audit every WMEval Lite evaluation item against its Unity reference.

Items: the 32 core cases (``test_core``) and the synthetic trajectories
(``test_full``). For each item the audit checks

1. action <-> reference: the Unity reference visibly performs the commanded
   actions (direction, magnitude, order, timing);
2. prompt <-> reference: the prompt's scene description and action text match
   the conditioning frame and the video;
3. meaningful evaluation: each declared (scored) dimension can produce a
   meaningful score on this item (tracked object moves and is visible, a
   contact exists for LPIPS, a causal chain for CausalStep, anchors visible
   for object consistency, slosh for fluid, shadows for lighting, ...);
4. reliable reference: no physically implausible / broken outcome, no
   rendering glitch or frozen frames;

with two independent sources:

* ``gt``  -- Unity ground truth (object transforms, camera poses, intrinsics,
  contact events, applicability opt-outs) and the video itself (frozen runs),
  per family, with explicit thresholds (``GT_*`` below);
* ``vlm`` -- a VLM judge (OpenAI-compatible endpoint; default NVIDIA
  inference ``azure/openai/gpt-6-astra``) shown the full-resolution
  conditioning frame plus frames spanning the whole reference, the prompt, the
  commanded action timeline and what each declared dimension needs; it
  returns a structured JSON verdict per check.

Every judge call's token usage is appended to ``<out>/vlm_usage.jsonl``. One
JSON result per item goes to ``<out>/items/``; ``summary`` merges them into
``<out>/summary.json`` and (``--publish``) the tracked
``lite/audit/{audit_results.jsonl,audit_summary.json}``.

    python tools/dataset/audit_dataset.py run --split all --round r1 --workers 8
    python tools/dataset/audit_dataset.py run --split synthetic --only 'cue_ball' --round r2 --force
    python tools/dataset/audit_dataset.py summary --round r1 [--publish]
    python tools/dataset/audit_dataset.py gt --only cue_ball            # GT checks only, printed

The key is read from ``--key-file`` (default ``~/.credentials/nv_vlm_api_key``)
and never printed or written.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUITE_ROOT = ROOT / "lite"
OUT_ROOT = ROOT / "runs/dataset_audit"
PUBLISHED = SUITE_ROOT / "audit"
DEFAULT_MODEL = "azure/openai/gpt-6-astra"
DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"
PROMPT_VERSION = "audit_v3"

# ---------------------------------------------------------------- GT thresholds
GT_MIN_TRACKED_DISP_M = 0.15        # kinematics: tracked object must move at least this much
GT_MIN_VISIBLE_MOVING_FRAMES = 30   # ... while visible in the frame for at least this many frames (1 s at 30 fps)
GT_MIN_TARGET_DISP_M = 0.3          # controllability: controlled object must move this much
GT_MIN_VISIBLE_FRAC = 0.75          # controlled object visible in at least this share of frames
GT_MIN_VISIBLE_FRAC_BY_FAMILY = {"ball_force": 0.95}
GT_BALL_RADIUS_M = {"white ball": 0.48, "mirror ball": 0.5}   # ball_force: the whole ball must stay in frame
GT_MIN_FULLY_IN_VIEW_FRAC = 0.97   # fixed camera: the ball stays in view (flow EPE needs it)
GT_FROZEN_PIXEL_DELTA = 6.0         # a pixel "changes" between consecutive frames above this |diff| (0-255)
GT_FROZEN_CHANGED_FRAC = 2e-4       # frames with fewer changed pixels than this share are "identical"
GT_MAX_FROZEN_RUN_SEC = 2.5         # longest identical run allowed while actions are still commanded
GT_FALL_M = 0.5                     # drop below the start height that counts as falling off/through
GT_MIN_SLOSH_STD = 0.004            # liquid-mask area std (fraction of frame) for a visible slosh
GT_ROAD_HALF_WIDTH_M = 10.5         # RacingCar: the roadside barriers stand at |x| ~ 12 m
GT_LANE_MIN_M, GT_LANE_MAX_M = 1.5, 9.0   # lateral offset of a performed lane change
GT_LANE_MAX_HEADING_DEG = 8.0       # ... after which the car is parallel to the road again (else it keeps drifting)
GT_AFTER_LANE_MAX_DRIFT_M = 2.0     # sideways drift while driving straight right after a lane change
GT_STEER_MIN_DEG = 3.0              # a steer step turns the car at least this much the commanded way
GT_MAX_BOTTLE_TILT_DEG = 3.0        # the sealed bottle stays upright
GT_TELEPORT_M_PER_FRAME = {"cue_ball": 0.5}   # a ball jumping farther in one frame was reset by the scene
GT_CHAR_SPEED_MPS = {"walk": 2.0, "sprint": 5.3}   # ChessBoard mannequin (v2 references)
GT_CHAR_MIN_SPEED_FRAC = 0.4        # a walk/sprint step covers at least this share of its nominal distance
GT_DUPLICATE_TRACK_M = 0.02         # two cue_ball references whose tracked ball paths differ by less are duplicates
GT_BEAM_MAX_INPUT_AFTER_FALL_SEC = 2.0   # core BeamDomino/causal keeps pushing 1.8 s after its ball leaves the platform

DIMENSION_NEEDS = {
    "lpips": "LPIPS (causal): the reference must contain at least one visible contact/collision event "
             "(ball hitting a wall, obstacle, pillar, domino, other ball) so collision-timing LPIPS is meaningful. "
             "(For the sealed liquid bottle no contact is expected; LPIPS falls back to temporal sampling there.)",
    "action_controllability": "Action controllability (optical-flow EPE vs the reference): the controlled object "
             "must visibly move in response to each commanded action and stay visible (not out of frame/occluded).",
    "prompt_scene_consistency": "Prompt-scene consistency: scored on the first frame of a generation conditioned "
             "on the conditioning frame and prompt; the scene description must match the conditioning frame.",
    "lighting_shading": "Lighting & shading (reference-free shadow tracking of {shadow}): the anchor object (the "
             "table) and its cast shadow must stay clearly visible throughout; other objects may leave the view at "
             "times.",
    "environment_object_consistency": "Environment/object consistency: anchor objects ({eoc}) must be visible in "
             "the first frame (they are detected there and tracked); camera motion that takes them out of view and "
             "back tests object permanence; the character ({char}) must be visible if one is configured.",
    "kinematics_mechanics": "Kinematics & mechanics: the tracked object(s) ({kin}) must visibly move in the "
             "reference and remain visible for a substantial part of their motion.",
    "physical_causality": "Physical causality (CausalStep QA): the reference must show a causal chain "
             "(ball strikes a pillar/structure or dominoes; roof tips, gold ball drops, dominoes cascade).",
    "fluid_mechanics": "Fluid mechanics: the bottle and the liquid inside must be clearly visible; bottle motion "
             "must produce visible liquid sloshing (except a pure rest control clip).",
}


# ---------------------------------------------------------------- items
@dataclass
class Item:
    item_id: str
    kind: str                      # core | synthetic
    family: str
    scene: str
    root: Path
    prompt: str
    perspective: str | None
    dimensions: list[str]
    not_applicable: dict[str, str]
    video: Path | None
    first_frame: Path | None
    artifacts: Path
    module_params: dict[str, Any]
    action_text: str = ""
    action_steps: list[dict[str, Any]] = field(default_factory=list)
    conditioning_case: str | None = None
    duration_sec: float | None = None


def _suite():
    from evaluation.lite.suite_loader import load_suite

    return load_suite(SUITE_ROOT)


def _timeline(traj_path: Path) -> str:
    """Plain-text commanded control timeline from a schema-v1 trajectory.yaml."""
    import yaml

    if not traj_path.is_file():
        return ""
    d = yaml.safe_load(traj_path.read_text(encoding="utf-8"))
    lines = []
    if d.get("description"):
        lines.append(str(d["description"]).strip())
    segs = d.get("segments") or []
    for s in segs[:40]:
        vals = {k: v for k, v in (s.get("values") or {}).items() if v not in (0, 0.0)}
        lines.append(f"  {float(s['start_sec']):.2f}-{float(s['end_sec']):.2f} s: " +
                     (", ".join(f"{k}={v:g}" for k, v in vals.items()) if vals else "no input"))
    if len(segs) > 40:
        lines.append(f"  ... ({len(segs) - 40} more segments)")
    return "\n".join(lines)


def _core_video(root: Path) -> Path | None:
    import yaml

    case = yaml.safe_load((root / "case.yaml").read_text(encoding="utf-8"))
    ref = case.get("reference") or {}
    if ref.get("video") and (root / ref["video"]).is_file():
        return root / ref["video"]
    vids = sorted((root / "reference").glob("*.mp4"))
    return vids[0] if vids else None


def load_items(split: str = "all") -> list[Item]:
    from evaluation.lite.case_loader import load_case
    from evaluation.lite.synthetic import load_synthetic_cases

    suite = _suite()
    items: list[Item] = []
    if split in ("all", "core"):
        dims_by_case: dict[str, list[str]] = {}
        for dim_id, dim in suite.dimensions.items():
            for c in dim.cases:
                dims_by_case.setdefault(c, []).append(dim_id)
        for case_id, dims in sorted(dims_by_case.items()):
            case = load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids())
            root = case.root
            ff = case.first_frame if case.first_frame and Path(case.first_frame).is_file() else None
            if ff is None and (root / "reference/ref.png").is_file():
                ff = root / "reference/ref.png"
            items.append(Item(
                item_id=case_id, kind="core", family="core:" + case_id.split("/")[0], scene=case.scene, root=root,
                prompt=case.prompt or "", perspective=case.perspective, dimensions=dims, not_applicable={},
                video=_core_video(root), first_frame=ff, artifacts=root / "reference/artifacts",
                module_params={k: dict(v) for k, v in case.module_params.items()},
                action_text=_timeline(root / "reference/artifacts/trajectory.yaml"),
            ))
    if split in ("all", "synthetic"):
        for sid, entry in sorted(load_synthetic_cases(suite).items()):
            t, case = entry.trajectory, entry.case
            meta = json.loads((case.root / "meta.json").read_text(encoding="utf-8"))
            ff = SUITE_ROOT / t.first_frame if t.first_frame else None
            if ff is not None and not ff.suffix.lower() in (".png", ".jpg"):
                ff = case.root / "reference/ref.png"
            items.append(Item(
                item_id=sid, kind="synthetic", family=t.family, scene=t.scene, root=case.root, prompt=t.prompt,
                perspective=t.perspective, dimensions=list(entry.dimensions),
                not_applicable=dict(entry.not_applicable), video=case.root / "reference/reference.mp4",
                first_frame=ff, artifacts=case.root / "reference/artifacts",
                module_params={k: dict(v) for k, v in case.module_params.items()},
                action_text=meta.get("action_description", ""), action_steps=meta.get("action_steps", []),
                conditioning_case=t.conditioning_case, duration_sec=meta.get("duration_sec"),
            ))
    return items


# ---------------------------------------------------------------- video helpers
def video_info(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"frames": n, "fps": fps, "width": w, "height": h, "duration_sec": n / fps if fps else None}


def frozen_runs(path: Path) -> dict[str, Any]:
    """Longest run of (near-)identical consecutive frames and per-second motion energy."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    prev, run, best, best_end, diffs = None, 0, 0, 0, []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            ad = np.abs(g - prev)
            d = float(np.mean(ad))
            diffs.append(d)
            # identical = (almost) no pixel changes; a small moving object still changes some pixels
            run = run + 1 if float(np.mean(ad > GT_FROZEN_PIXEL_DELTA)) < GT_FROZEN_CHANGED_FRAC else 0
            if run > best:
                best, best_end = run, i
        prev = g
        i += 1
    cap.release()
    diffs_a = np.asarray(diffs) if diffs else np.zeros(1)
    per_sec = [round(float(diffs_a[k:k + int(fps)].mean()), 3) for k in range(0, len(diffs_a), max(1, int(fps)))]
    return {"longest_identical_run_sec": round(best / fps, 2),
            "longest_identical_run_end_sec": round(best_end / fps, 2),
            "mean_abs_frame_diff": round(float(diffs_a.mean()), 3), "motion_per_sec": per_sec}


def sample_frames(path: Path, n: int, width: int, extra_times: list[float] | None = None) -> list[tuple[float, bytes]]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    idx = set(np.linspace(0, N - 1, n).round().astype(int).tolist())
    for t in extra_times or []:
        idx.add(int(min(N - 1, max(0, round(t * fps)))))
    out = []
    for i in sorted(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            continue
        h = int(round(f.shape[0] * width / f.shape[1]))
        f = cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append((i / fps, buf.tobytes()))
    cap.release()
    return out


def image_bytes(path: Path, max_width: int = 1280) -> bytes:
    import cv2

    f = cv2.imread(str(path))
    if f.shape[1] > max_width:
        f = cv2.resize(f, (max_width, int(f.shape[0] * max_width / f.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


# ---------------------------------------------------------------- GT helpers
def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _objects(art: Path) -> dict[str, np.ndarray]:
    p = art / "object_transforms.json"
    if not p.is_file():
        return {}
    d = _load_json(p)
    out = {}
    for k, v in (d.get("objects") or {}).items():
        pos = v.get("positions_world") if isinstance(v, dict) else None
        if pos:
            out[k] = np.asarray(pos, dtype=np.float64)
    return out


def _camera(art: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    P = np.load(art / "poses.npy") if (art / "poses.npy").is_file() else None
    for name in ("intrinsics.npy", "intrinsics_t4.npy"):
        if (art / name).is_file():
            return P, np.load(art / name)
    return P, None


def _project(P: np.ndarray, K: np.ndarray, pts: np.ndarray, wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame pixel coords and in-view flags of world points (OpenCV-style c2w, any handedness)."""
    T = min(len(pts), len(P))
    uv = np.full((T, 2), np.nan)
    vis = np.zeros(T, dtype=bool)
    for i in range(T):
        c2w = P[i]
        q = np.linalg.inv(c2w) @ np.r_[pts[i], 1.0]
        k = K[i] if K.ndim == 2 and K.shape[1] == 4 else (K if K.ndim == 1 else np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
        if q[2] <= 0.05:
            continue
        u, v = k[0] * q[0] / q[2] + k[2], k[1] * q[1] / q[2] + k[3]
        uv[i] = (u, v)
        vis[i] = 0 <= u < wh[0] and 0 <= v < wh[1]
    return uv, vis


def _wh(art: Path, K: np.ndarray | None) -> tuple[int, int]:
    if K is not None:
        k = K[0] if K.ndim == 2 and K.shape[1] == 4 else None
        if k is not None:
            return int(round(k[2] * 2)), int(round(k[3] * 2))
    return 1280, 720


def _flag(flags: list, code: str, severity: str, detail: str) -> None:
    flags.append({"code": code, "severity": severity, "detail": detail})


def _commanded_end_sec(item: Item) -> float:
    """Time of the last non-zero commanded control (synthetic action.npz)."""
    p = item.root / "action.npz"
    if not p.is_file():
        return item.duration_sec or 0.0
    z = np.load(p, allow_pickle=False)
    ts, vals = z["timestamps_sec"], np.asarray(z["values"], dtype=np.float64)
    nz = np.where(np.abs(vals).sum(axis=1) > 0)[0]
    if not len(nz):
        return 0.0
    dt = float(ts[1] - ts[0]) if len(ts) > 1 else 0.0
    return float(ts[nz[-1]] + dt)


def _disp_stats(pos: np.ndarray, fps: float) -> dict[str, Any]:
    d = np.linalg.norm(pos[:, [0, 2]] - pos[0, [0, 2]], axis=1)
    sp = np.r_[0.0, np.linalg.norm(np.diff(pos[:, [0, 2]], axis=0), axis=1) * fps]
    return {"max_disp_m": round(float(d.max()), 3), "final_disp_m": round(float(d[-1]), 3),
            "path_m": round(float(np.sum(np.linalg.norm(np.diff(pos[:, [0, 2]], axis=0), axis=1))), 3),
            "min_dy_m": round(float(pos[:, 1].min() - pos[0, 1]), 3), "speed": sp}


def gt_checks(item: Item) -> dict[str, Any]:
    flags: list[dict[str, str]] = []
    stats: dict[str, Any] = {}
    art = item.artifacts
    if item.video is None or not item.video.is_file():
        _flag(flags, "no_reference_video", "major", "reference video missing")
        return {"flags": flags, "stats": stats}
    vi = video_info(item.video)
    stats["video"] = vi
    fr = frozen_runs(item.video)
    stats["frozen"] = fr
    fam = item.family
    objs = _objects(art)
    P, K = _camera(art)
    wh = _wh(art, K)
    fps_gt = None
    if (art / "unity_gt.json").is_file():
        g = _load_json(art / "unity_gt.json")
        fps_gt = float(g.get("fps") or 0) if isinstance(g, dict) else None
    fps_gt = fps_gt or (vi["fps"] if fam != "liquid_container" else 30.0)

    # frozen video while actions are still commanded (synthetic) -- or anywhere in physics clips
    if item.kind == "synthetic":
        cmd_end = _commanded_end_sec(item)
        stats["commanded_end_sec"] = round(cmd_end, 2)
        run, end = fr["longest_identical_run_sec"], fr["longest_identical_run_end_sec"]
        active = False
        if (item.root / "action.npz").is_file():
            z = np.load(item.root / "action.npz", allow_pickle=False)
            ts, vals = z["timestamps_sec"], np.abs(np.asarray(z["values"], dtype=np.float64)).sum(axis=1)
            win = (ts >= end - run + 0.2) & (ts <= end - 0.2)
            active = bool(win.any() and (vals[win] > 0).mean() > 0.5)
        if run > GT_MAX_FROZEN_RUN_SEC and active:
            _flag(flags, "frozen_during_actions", "minor",
                  f"{run:.1f} s of identical frames ending at {end:.1f} s while controls are commanded until {cmd_end:.1f} s")

    def tracked(label: str, need_motion: bool, need_vis: bool, who: str, dim: str) -> None:
        if label not in objs:
            _flag(flags, "gt_missing_object", "minor", f"{label!r} not in object_transforms.json")
            return
        pos = objs[label]
        st = _disp_stats(pos, fps_gt)
        sp = st.pop("speed")
        rec = {k: v for k, v in st.items()}
        if P is not None and K is not None:
            uv, vis = _project(P, K, pos, wh)
            moving = sp[: len(vis)] > 0.05
            rec["visible_frac"] = round(float(vis.mean()), 3)
            radius = GT_BALL_RADIUS_M.get(label)
            if radius is not None and need_vis:
                inside = []
                for i in range(len(vis)):
                    q = np.linalg.inv(P[i]) @ np.r_[pos[i], 1.0]
                    k = K[i] if K.ndim == 2 else K
                    r_px = k[0] * radius / q[2] if q[2] > 0 else np.inf
                    u, v = uv[i]
                    inside.append(bool(q[2] > 0 and r_px <= u <= wh[0] - r_px and r_px <= v <= wh[1] - r_px))
                rec["fully_in_view_frac"] = round(float(np.mean(inside)), 3)
                if np.mean(inside) < GT_MIN_FULLY_IN_VIEW_FRAC:
                    _flag(flags, f"{who}_clipped_by_frame", "major",
                          f"{label} is cut by the image border in {1 - np.mean(inside):.0%} of frames")
            rec["visible_moving_frames"] = int(np.sum(vis & moving))
            rec["moving_frames"] = int(np.sum(moving))
            if need_vis and rec["visible_frac"] < GT_MIN_VISIBLE_FRAC_BY_FAMILY.get(fam, GT_MIN_VISIBLE_FRAC) \
                    and fam != "liquid_container":
                _flag(flags, f"{who}_mostly_out_of_view", "major",
                      f"{label} visible in {rec['visible_frac']:.0%} of frames ({dim})")
        stats[f"obj:{label}"] = rec
        if need_motion:
            thr = GT_MIN_TRACKED_DISP_M if who == "tracked" else GT_MIN_TARGET_DISP_M
            if st["max_disp_m"] < thr:
                _flag(flags, f"{who}_never_moves", "major",
                      f"{label} max displacement {st['max_disp_m']:.3f} m < {thr} m ({dim})")
            elif who == "tracked" and "visible_moving_frames" in rec and rec["visible_moving_frames"] < GT_MIN_VISIBLE_MOVING_FRAMES:
                _flag(flags, "tracked_motion_not_visible", "major",
                      f"{label} moves {st['max_disp_m']:.2f} m but only {rec['visible_moving_frames']} moving frames are "
                      f"in view ({dim})")
        jump = GT_TELEPORT_M_PER_FRAME.get(fam)
        if jump is not None and len(pos) > 1:
            steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            if steps.max() > jump:
                _flag(flags, "scene_reset", "major",
                      f"{label} jumps {steps.max():.2f} m in one frame at {int(np.argmax(steps)) + 1} (scene reset)")
        if st["min_dy_m"] < -GT_FALL_M and fam != "beam_domino":   # BeamDomino's ball leaving the platform: see falls_while_commanded
            _flag(flags, f"{who}_falls", "major", f"{label} drops {st['min_dy_m']:.2f} m below its start height")

    if item.kind == "synthetic":
        if fam == "cue_ball":
            for lab in [s.strip() for s in str(item.module_params.get("kinematics_mechanics", {}).get("kin_text_prompt", "")).split(".") if s.strip()]:
                tracked(lab, True, False, "tracked", "kinematics_mechanics")
            tracked("white ball", True, False, "target", "action")
        elif fam == "ego_vehicle":
            tracked("car", True, True, "tracked", "kinematics_mechanics")
            g = _load_json(art / "unity_gt.json")
            rows = g["frames"]
            ctrl = np.asarray([r["control"] for r in rows], dtype=np.float64)
            vel = np.linalg.norm(np.asarray([r["car_velocity"] for r in rows])[:, [0, 2]], axis=1)
            stuck = (ctrl[:, 1] > 0.25) & (vel < 0.3)
            # ignore the first 0.5 s of any throttle burst (spin-up)
            run, worst = 0, 0
            for s in stuck:
                run = run + 1 if s else 0
                worst = max(worst, run)
            stats["car_max_speed_mps"] = round(float(vel.max()), 2)
            stats["car_longest_stuck_sec"] = round(worst / fps_gt, 2)
            if worst / fps_gt > 1.0:
                _flag(flags, "car_stuck", "major", f"throttle on but speed < 0.3 m/s for {worst / fps_gt:.1f} s")
            pos = np.asarray([r["car_pos"] for r in rows], dtype=np.float64)
            stats["car_max_abs_x_m"] = round(float(np.abs(pos[:, 0]).max()), 2)
            if np.abs(pos[:, 0]).max() > GT_ROAD_HALF_WIDTH_M:
                k = int(np.argmax(np.abs(pos[:, 0]) > GT_ROAD_HALF_WIDTH_M))
                _flag(flags, "car_hits_barrier", "major",
                      f"car reaches the roadside barrier (|x| > {GT_ROAD_HALF_WIDTH_M} m) at {k / fps_gt:.1f} s")
            vel3 = np.asarray([r["car_velocity"] for r in rows], dtype=np.float64)

            def heading(i: int) -> float:
                for j in range(i, -1, -1):   # last frame with a usable velocity (a stopped car keeps its heading)
                    if np.hypot(vel3[j, 0], vel3[j, 2]) > 0.5:
                        return math.degrees(math.atan2(vel3[j, 0], vel3[j, 2]))
                return float("nan")

            t, checks = 0.0, []
            for st in item.action_steps:
                sec = float(st.get("sec", 0))
                a, b = int(round(t * fps_gt)), min(len(rows) - 1, int(round((t + sec) * fps_gt)))
                t += sec
                sign = {"left": -1, "right": 1}.get(st.get("dir"))
                if st["kind"] == "lane":
                    dx, dh = pos[b, 0] - pos[a, 0], heading(b) - heading(a)
                    ok = sign * dx >= GT_LANE_MIN_M and abs(dx) <= GT_LANE_MAX_M and abs(dh) <= GT_LANE_MAX_HEADING_DEG
                    checks.append(f"lane {st['dir']} {sec:g}s: dx={dx:+.1f} m dheading={dh:+.0f}")
                    if not ok:
                        _flag(flags, "lane_change_not_performed", "major",
                              f"lane change {st['dir']} at {a / fps_gt:.1f}-{b / fps_gt:.1f} s: lateral {dx:+.1f} m, "
                              f"heading change {dh:+.0f} deg (want {GT_LANE_MIN_M}-{GT_LANE_MAX_M} m, |dh| <= "
                              f"{GT_LANE_MAX_HEADING_DEG})")
                elif st["kind"] in ("drive", "coast") and checks and checks[-1].startswith("lane"):
                    dx = pos[b, 0] - pos[a, 0]
                    checks.append(f"{st['kind']} after lane {sec:g}s: dx={dx:+.1f} m")
                    if abs(dx) > GT_AFTER_LANE_MAX_DRIFT_M:
                        _flag(flags, "lane_change_not_performed", "major",
                              f"car drifts {dx:+.1f} m sideways while driving straight at {a / fps_gt:.1f}-{b / fps_gt:.1f} s "
                              f"after a lane change (it did not end parallel to the road)")
                elif st["kind"] == "steer":
                    dh = heading(b) - heading(a)
                    checks.append(f"steer {st['dir']} {sec:g}s: dheading={dh:+.0f}")
                    if not (sign * dh >= GT_STEER_MIN_DEG):
                        _flag(flags, "steer_not_performed", "major",
                              f"steer {st['dir']} at {a / fps_gt:.1f}-{b / fps_gt:.1f} s changes heading by {dh:+.0f} deg")
            last = item.action_steps[-1] if item.action_steps else None
            if last is not None and last["kind"] == "lane" and t < len(rows) / fps_gt - 0.5:
                # the clip ends coasting right after a lane change: the car must not drift back
                a, b = int(round(t * fps_gt)), min(len(rows) - 1, int(round((t + 3.0) * fps_gt)))
                dx = pos[b, 0] - pos[a, 0]
                checks.append(f"coast after lane: dx={dx:+.1f} m")
                if abs(dx) > GT_AFTER_LANE_MAX_DRIFT_M:
                    _flag(flags, "lane_change_not_performed", "major",
                          f"car drifts {dx:+.1f} m sideways while coasting at {a / fps_gt:.1f}-{b / fps_gt:.1f} s after "
                          f"a lane change (it did not end parallel to the road)")
            stats["car_steps"] = checks
        elif fam == "ball_force":
            label = "white ball" if "white ball" in objs else "mirror ball"
            tracked(label, True, True, "target", "action_controllability")
        elif fam == "beam_domino":
            tracked("FloorBall", True, False, "target", "action_controllability")
            seg = art / "causal_segments.json"
            if seg.is_file():
                s = _load_json(seg)
                stats["causal_shape"] = s.get("trajectory_shape")
                stats["causal_events"] = [e.get("name") for e in s.get("events", [])]
            fb = objs.get("FloorBall")
            if fb is not None:
                below = fb[:, 1] < fb[0, 1] - GT_FALL_M
                fall = float(np.argmax(below)) / fps_gt if below.any() else None
                stats["ball_leaves_platform_sec"] = None if fall is None else round(fall, 2)
                cmd_end = _commanded_end_sec(item)
                if fall is not None and cmd_end - fall > GT_BEAM_MAX_INPUT_AFTER_FALL_SEC:
                    _flag(flags, "falls_while_commanded", "major",
                          f"ball leaves the platform at {fall:.1f} s but inputs continue until {cmd_end:.1f} s")
            moved = [k for k, v in objs.items() if k.startswith("Domino") and np.linalg.norm(v[-1] - v[0]) > 0.05]
            stats["dominoes_moved"] = len(moved)
            if "Roof" in objs:
                stats["roof_moved_m"] = round(float(np.linalg.norm(objs["Roof"][-1] - objs["Roof"][0])), 3)
            if "physical_causality" in item.dimensions and not stats.get("causal_shape"):
                _flag(flags, "no_causal_chain", "major", "scored on physical_causality but no causal segments")
            if "physical_causality" in item.not_applicable and moved:
                _flag(flags, "opted_out_but_chain", "minor", f"{len(moved)} dominoes moved but causality is opted out")
        elif fam == "character":
            tracked("person", True, True, "target", "environment_object_consistency")
            pos = objs.get("person")
            if pos is not None:
                t, checks = 0.0, []
                for st in item.action_steps:
                    sec = {"jump": 1 / 30}.get(st["kind"], float(st.get("sec", 0)))
                    a, b = int(round(t * fps_gt)), min(len(pos) - 1, int(round((t + sec) * fps_gt)))
                    t += sec
                    if st["kind"] in GT_CHAR_SPEED_MPS:
                        dist = float(np.linalg.norm(pos[b, [0, 2]] - pos[a, [0, 2]]))
                        want = GT_CHAR_SPEED_MPS[st["kind"]] * sec
                        checks.append(f"{st['kind']} {st['dir']} {sec:g}s: {dist:.1f}/{want:.1f} m")
                        if dist < GT_CHAR_MIN_SPEED_FRAC * want:
                            _flag(flags, "character_move_not_performed", "major",
                                  f"{st['kind']} {st['dir']} at {a / fps_gt:.1f}-{b / fps_gt:.1f} s covers {dist:.1f} m "
                                  f"(nominal {want:.1f} m)")
                stats["character_steps"] = checks
        elif fam == "liquid_container":
            tracked("bottle", item.action_steps != [] and any(s["kind"] != "rest" for s in item.action_steps), False,
                    "target", "fluid_mechanics")
            g = _load_json(art / "unity_gt.json")
            rows = g["frames"]
            q = np.asarray([r["target_rot"] for r in rows], dtype=np.float64)
            upy = 1 - 2 * (q[:, 0] ** 2 + q[:, 2] ** 2)
            tilt = np.degrees(np.arccos(np.clip(upy, -1, 1)))
            stats["bottle_max_tilt_deg"] = round(float(tilt.max()), 2)
            if tilt.max() > GT_MAX_BOTTLE_TILT_DEG:
                _flag(flags, "bottle_tilts", "major", f"bottle tilts {tilt.max():.0f} deg (translation-only control)")
            yb = np.asarray([r["target_pos"][1] for r in rows])
            stats["bottle_end_dy_m"] = round(float(yb[-1] - yb[0]), 3)
            if abs(yb[-1] - yb[0]) > 0.05:
                _flag(flags, "bottle_hovers", "major", f"bottle ends {yb[-1] - yb[0]:+.2f} m from its rest height")
            lm = art / "liquid_mask.npz"
            if lm.is_file():
                z = np.load(lm, allow_pickle=False)
                key = [k for k in z.files if "mask" in k.lower()] or z.files
                m = z[key[0]]
                if m.ndim == 3:
                    area = m.reshape(len(m), -1).mean(axis=1)
                    stats["liquid_area_mean"] = round(float(area.mean()), 4)
                    stats["liquid_area_std"] = round(float(area.std()), 5)
                    stats["liquid_empty_frames"] = int(np.sum(area < 1e-4))
                    if np.sum(area < 1e-4) > 0.1 * len(area):
                        _flag(flags, "liquid_mask_missing", "major", f"liquid mask empty on {int(np.sum(area < 1e-4))} frames")
        elif fam == "camera_yaw":
            if P is not None:
                fwd = P[:, :3, 2]
                yaw = np.degrees(np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2])))
                stats["yaw_range_deg"] = round(float(yaw.max() - yaw.min()), 1)
                stats["yaw_final_deg"] = round(float(yaw[-1] - yaw[0]), 1)
            eoc = item.module_params.get("environment_object_consistency", {})
            anchors = {}
            for lab, pos in objs.items():
                if P is None or K is None:
                    break
                _, vis = _project(P, K, pos, wh)
                left = bool(vis[0]) and not vis.all()
                returned = left and bool(vis[np.argmin(vis) :].any()) if left else False
                anchors[lab] = {"visible_frame0": bool(vis[0]), "visible_frac": round(float(vis.mean()), 3),
                                "leaves_view": bool(vis[0] and not vis.all()),
                                "returns": bool(vis[0] and (~vis).any() and vis[int(np.argmax(~vis)):].any())}
            stats["anchors"] = anchors
            stats["eoc_text_prompt"] = eoc.get("eoc_text_prompt")
        elif fam == "camera_orbit":
            pass
    return {"flags": flags, "stats": stats}


# ---------------------------------------------------------------- VLM judge
_USAGE_LOCK = threading.Lock()

SYSTEM = (
    "You are auditing a benchmark dataset for action-conditioned video world models. Each item pairs a text prompt "
    "(a scene description, usually followed by an action description), a conditioning first frame and a Unity "
    "reference video rendered by replaying the commanded actions in the Unity scene. World-model generations are "
    "later scored against this reference on the listed dimensions. Your job: decide whether the item is sensible and "
    "yields a meaningful evaluation. Be concrete and conservative: report a MAJOR issue only when you can see it "
    "clearly in the frames (or the text is clearly contradicted); report doubtful or cosmetic points as MINOR. "
    "Do not penalise approximate distances/speeds in the prompt unless clearly wrong (e.g. factor >1.5), and do not "
    "treat stylistic wording as an issue. Answer in JSON only."
)

SCHEMA_HINT = """Return a JSON object with exactly these keys:
{
  "first_frame_description": "<2-3 sentences: what the conditioning frame shows (objects, colours, camera viewpoint, lighting/sun)>",
  "video_description": "<3-6 sentences: what happens over time in the reference video, with approximate times>",
  "prompt_claims": [{"claim": "<a factual claim of the prompt: object, count, colour, layout, camera, lighting, or what moves/rotates/happens>", "status": "supported"|"contradicted"|"never_exhibited"|"unverifiable", "note": "..."}],
  "action_match": {"ok": true|false, "severity": "none"|"minor"|"major", "issues": ["..."]},
  "prompt_match": {"ok": true|false, "severity": "none"|"minor"|"major", "issues": ["<each concrete contradiction between the prompt text and the first frame / video>"]},
  "dimensions": {"<dimension id>": {"meaningful": true|false, "severity": "none"|"minor"|"major", "reason": "..."}},
  "reference_reliable": {"ok": true|false, "severity": "none"|"minor"|"major", "issues": ["<physically implausible/broken outcomes, glitches, frozen frames, objects through walls/floor, vehicle off road or stuck, character falling off or through objects>"]},
  "duplicate_or_trivial": {"flag": true|false, "reason": "..."},
  "verdict": "ok"|"problem",
  "summary": "<one sentence>"
}
action_match: does the reference visibly perform the commanded actions (direction, magnitude, ordering, timing)? If nothing visible happens for an action that should have a visible effect, that is MAJOR. If the item has no commanded actions beyond a fixed camera move, judge the camera move.
prompt_claims: enumerate ALL concrete claims of the prompt (up to ~25), including generic capability statements such as "it rotates in response to control inputs" -- mark those "never_exhibited" when the commanded actions/video never do it (that misleads a world model and is a prompt issue), "contradicted" when the frames show otherwise (wrong colour/count/object/camera height or distance/sun elevation/time of day), and also flag internal contradictions between two sentences of the prompt.
prompt_match: compare every factual claim of the prompt (objects, counts, colours, layout, camera perspective/height/distance, lighting/sun direction and elevation, claims about what moves or rotates) with the first frame and the video.
A "contradicted" or "never_exhibited" claim must also appear in prompt_match.issues (MAJOR when it states something about the scene or the motion that is clearly false for this item; MINOR for small approximations).
verdict = "problem" iff any MAJOR issue."""


def _dimension_text(item: Item) -> str:
    mp = item.module_params
    shadow = mp.get("shadow_consistency", {}).get("shadow_text_prompt", "table, chairs, lamp")
    eoc = mp.get("environment_object_consistency", {})
    kin = mp.get("kinematics_mechanics", {}).get("kin_text_prompt", "")
    char = eoc.get("eoc_character_prompt", "")
    if "no_character" in str(char):
        char = "none configured"
    lines = []
    for d in item.dimensions:
        lines.append(f"- {d}: " + DIMENSION_NEEDS.get(d, d).format(shadow=shadow, eoc=eoc.get("eoc_text_prompt", ""),
                                                              char=char, kin=kin))
    for d, why in item.not_applicable.items():
        lines.append(f"- ({d} is declared NOT applicable for this item: {why}; do not judge it)")
    return "\n".join(lines)


def _family_notes(item: Item) -> str:
    notes = {
        "cue_ball": "The camera rides on the white cue ball at ball level (first person), keeping a fixed heading; "
                    "the white cue ball itself is just below the bottom edge of the frame and is not visible. "
                    "Strikes apply impulses to the cue ball (forward = the initial camera heading).",
        "ego_vehicle": "Third-person chase camera behind the car on a race track.",
        "ball_force": "Fixed camera; forces push the controlled ball. The action text names directions as seen "
                      "from the camera (forward = away from the camera, right = screen right); a force adds to the "
                      "ball's existing momentum, so after a direction change the path curves rather than turning "
                      "instantly, and collisions deflect it.",
        "beam_domino": "The blue ball is pushed by forces (forward = toward the beam structure); the scene camera follows the ball, also when it rolls off the edge of the floating platform and falls (as in the core BeamDomino recordings).",
        "character": "Third-person Cinemachine camera following the small gray mannequin at a fixed offset (as in "
                     "the core ChessBoard/walk_third recording), so walking away from the camera shows mainly as the "
                     "statues coming closer (parallax), not as the character shrinking. Known scene limitation, "
                     "shared with the core recording: when the follow camera passes through a giant foreground statue "
                     "that statue is clipped away by the near plane; report it as MINOR.",
        "liquid_container": "Side camera that travels with the bottle (as in the core LiquidContainer recordings), so the floor scrolls while the bottle stays near the centre; the sealed bottle translates (slides/shakes/jolts) without rotating.",
        "camera_yaw": "Only the camera rotates (yaw) in place or around the subject; nothing else is commanded.",
        "camera_orbit": "The camera orbits around the table keeping it centred; the sun is fixed.",
    }
    if item.kind == "core":
        return ("This is a curated core case (paper recording). The commanded control timeline is given below; "
                "for camera cases the camera motion is the action.")
    return notes.get(item.family, "")


def _extra_times(item: Item) -> list[float]:
    """Denser frames around short commanded events (strikes, jolts, jumps)."""
    out: list[float] = []
    t = 0.0
    for s in item.action_steps:
        k = s.get("kind")
        sec = float(s.get("sec", 0) or 0)
        if k in ("strike", "repeat", "jolt", "jump"):
            for dt in (0.1, 0.3, 0.6, 1.0):
                out.append(t + dt)
        if k == "strike":
            sec = 0.5
        elif k == "repeat":
            sec = 0.5 * (2 if s.get("n") == "twice" else 3)
        elif k == "jolt":
            sec = 0.25
        elif k == "jump":
            sec = 1 / 30
        elif k == "sweep":
            sec = 2 * float(s["amp"]) / float(s["speed"])
        elif k == "look_back":
            sec = 180 / float(s["speed"])
        elif k == "lane":
            sec = float(s["sec"])
        elif k in ("shake", "chirp"):
            sec = float(s["sec"])
        t += sec
    return out[:16]


def vlm_check(item: Item, client, model: str, usage_log: Path, n_frames: int = 28, frame_width: int = 448,
              retries: int = 3) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    header = [
        f"ITEM: {item.item_id} ({item.kind}; family {item.family}; perspective {item.perspective})",
        f"FAMILY NOTES: {_family_notes(item)}",
        "PROMPT (exactly as given to world models):\n<<<\n" + item.prompt.strip() + "\n>>>",
    ]
    if item.kind == "core":
        header.append("COMMANDED CONTROL TIMELINE:\n" + (item.action_text or "(none recorded)"))
    else:
        header.append("ACTION DESCRIPTION (its compiled controls drove the Unity replay):\n" + item.action_text)
    header.append("DIMENSIONS THIS ITEM IS SCORED ON AND WHAT EACH NEEDS:\n" + _dimension_text(item))
    header.append(SCHEMA_HINT)
    content.append({"type": "text", "text": "\n\n".join(header)})
    if item.first_frame is not None and Path(item.first_frame).is_file():
        content.append({"type": "text", "text": "CONDITIONING FIRST FRAME (full resolution):"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes(Path(item.first_frame))).decode(),
            "detail": "high"}})
    frames = sample_frames(item.video, n_frames, frame_width, _extra_times(item))
    content.append({"type": "text", "text": f"REFERENCE VIDEO FRAMES ({len(frames)} frames, timestamps in seconds):"})
    for t, b in frames:
        content.append({"type": "text", "text": f"t={t:.2f}s"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}})
    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                response_format={"type": "json_object"},
            )
            u = resp.usage
            rec = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "item": item.item_id,
                   "model": model, "prompt_version": PROMPT_VERSION,
                   "input_tokens": getattr(u, "prompt_tokens", None), "output_tokens": getattr(u, "completion_tokens", None),
                   "total_tokens": getattr(u, "total_tokens", None),
                   "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None),
                   "cached_input_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None),
                   "latency_sec": round(time.time() - t0, 1), "n_images": len(frames) + 1}
            with _USAGE_LOCK, usage_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            text = resp.choices[0].message.content or ""
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            data["_frames"] = [round(t, 2) for t, _ in frames]
            return data
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(3 * (attempt + 1))
    return {"error": last_err}


# ---------------------------------------------------------------- decision
def decide(gt: dict[str, Any], vlm: dict[str, Any] | None) -> dict[str, Any]:
    major = [f"gt:{f['code']}: {f['detail']}" for f in gt.get("flags", []) if f["severity"] == "major"]
    minor = [f"gt:{f['code']}: {f['detail']}" for f in gt.get("flags", []) if f["severity"] != "major"]
    if vlm and "error" not in vlm:
        for key in ("action_match", "prompt_match", "reference_reliable"):
            v = vlm.get(key) or {}
            sev = v.get("severity", "none")
            for issue in v.get("issues") or []:
                (major if sev == "major" else minor).append(f"vlm:{key}: {issue}")
        for d, v in (vlm.get("dimensions") or {}).items():
            if v.get("severity") == "major" or v.get("meaningful") is False:
                (major if v.get("severity") == "major" else minor).append(f"vlm:dimension:{d}: {v.get('reason')}")
            elif v.get("severity") == "minor":
                minor.append(f"vlm:dimension:{d}: {v.get('reason')}")
    elif vlm and "error" in vlm:
        minor.append(f"vlm:error: {vlm['error']}")
    return {"status": "problem" if major else ("review" if minor else "ok"), "major": major, "minor": minor}


def safe_name(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", item_id)


def audit_item(item: Item, out: Path, client, model: str, skip_vlm: bool, force: bool) -> dict[str, Any]:
    path = out / "items" / f"{safe_name(item.item_id)}.json"
    if path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    gt = gt_checks(item)
    vlm = None if skip_vlm else vlm_check(item, client, model, out / "vlm_usage.jsonl")
    rec = {"item_id": item.item_id, "kind": item.kind, "family": item.family, "conditioning_case": item.conditioning_case,
           "dimensions": item.dimensions, "not_applicable": item.not_applicable,
           "video": str(item.video.relative_to(ROOT)) if item.video else None,
           "video_sha256": _sha(item.video) if item.video else None,
           "prompt_sha256": _sha_text(item.prompt), "action_text": item.action_text,
           "gt": gt, "vlm": vlm, "decision": decide(gt, vlm), "model": None if skip_vlm else model,
           "prompt_version": PROMPT_VERSION, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=_np_default) + "\n", encoding="utf-8")
    return rec


def _np_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def _sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sha_text(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def usage_summary(log: Path) -> dict[str, Any]:
    tot = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_input_tokens": 0}
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tot["calls"] += 1
            for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens"):
                tot[k] += int(r.get(k) or 0)
    return tot


# Remaining major findings reviewed by hand and accepted (audit/README.md "Accepted as-is"): item regex -> reason.
ACCEPTED = {
    r"^BallRoller/ctrl$": "the recording's realized key timing is slower than the commanded 2.96 s script (the ball "
                          "is still driven until ~5.3 s); re-deriving it would change the paper-parity "
                          "controllability actions -- documented in audit/README.md",
    r"^BeamDomino/causal$": "paper recording; the cascade is verified from the Unity log and the recorded reference "
                            "judge answers (csr_unity 100); the follow camera leaves the structure",
    r"^ChessBoard/walk_third$": "follow camera clips through foreground statues (scene rig, as recorded)",
    r"^RacingCar/kinematics_third$": "paper recording; the car ends against the barrier from 6.3 s",
    r"^WanderingGarden/scene_(360|perturb)_out$|camera_yaw/WanderingGarden__scene_360_out/":
        "outward Table 7 view: the tree anchor is not in frame 0; lanterns/rocks are tracked",
    r"beam_domino/": "brief occlusion by the structure, a shape-specific (direct strike / glancing) chain judged "
                     "against the canonical one, or <= 2 s of pushing after the ball leaves the platform (as in "
                     "core BeamDomino/causal)",
    r"camera_orbit/": "lighting is reference-free: the Unity orbit's cropped floor shadow is not scored",
    r"character/": "small mannequin behind statues: forward walking shows only as parallax, a jump can be occluded",
    r"cue_ball/": "the yellow ball is in view for >= 1 s of its motion (ball-mounted camera with a fixed heading)",
}


def _accepted(item_id: str) -> str | None:
    return next((why for rx, why in ACCEPTED.items() if re.search(rx, item_id)), None)


def summarize(out: Path, publish: bool) -> dict[str, Any]:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "items").glob("*.json"))]
    by_family: dict[str, dict[str, int]] = {}
    for r in recs:
        if r["decision"]["status"] == "problem" and _accepted(r["item_id"]):
            r["decision"]["status"] = "accepted"
            r["decision"]["accepted_reason"] = _accepted(r["item_id"])
        fam = r["family"] if r["kind"] == "synthetic" else "core"
        c = by_family.setdefault(fam, {"ok": 0, "review": 0, "accepted": 0, "problem": 0})
        c[r["decision"]["status"]] += 1
    summary = {"schema_version": 1, "round": out.name, "items": len(recs),
               "by_status": {s: sum(1 for r in recs if r["decision"]["status"] == s)
                             for s in ("ok", "review", "accepted", "problem")},
               "by_family": by_family, "vlm_usage": usage_summary(out / "vlm_usage.jsonl"),
               "problems": {r["item_id"]: r["decision"]["major"] for r in recs if r["decision"]["status"] == "problem"},
               "accepted": {r["item_id"]: r["decision"]["accepted_reason"] for r in recs
                            if r["decision"]["status"] == "accepted"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if publish:
        PUBLISHED.mkdir(parents=True, exist_ok=True)
        with (PUBLISHED / "audit_results.jsonl").open("w", encoding="utf-8") as fh:
            for r in recs:
                slim = {k: r[k] for k in ("item_id", "kind", "family", "dimensions", "not_applicable", "video_sha256",
                                          "prompt_sha256", "decision", "model", "prompt_version", "utc")}
                slim["decision"] = {**r["decision"], "minor": [x[:240] for x in r["decision"]["minor"]],
                                    "major": [x[:240] for x in r["decision"]["major"]]}
                slim["gt_flags"] = r["gt"]["flags"]
                v = r.get("vlm") or {}
                slim["vlm"] = {k: v.get(k) for k in ("verdict", "summary")} if v else None
                fh.write(json.dumps(slim, ensure_ascii=False, default=_np_default) + "\n")
        first = OUT_ROOT / "r1" / "summary.json"
        if first.is_file() and out.name != "r1":
            r1 = json.loads(first.read_text(encoding="utf-8"))
            summary["round_1"] = {k: r1[k] for k in ("items", "by_status", "by_family")}
            summary["round_1"]["problems"] = {k: [x[:200] for x in v] for k, v in r1["problems"].items()}
        (PUBLISHED / "audit_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    return summary


# ---------------------------------------------------------------- plan screening
# GT failure codes that make a synthetic trajectory's plan unusable (its reference cannot give
# a meaningful score on a declared dimension, or is physically broken). A plan failing one is
# recorded in synthetic/screening.json and the generator swaps it for the next-best plan.
SCREEN_CODES = {
    "cue_ball": {"tracked_never_moves", "tracked_motion_not_visible", "scene_reset", "duplicate_tracked_path",
                 "target_falls", "tracked_falls", "near_duplicate_plan"},
    "ego_vehicle": {"car_stuck", "car_hits_barrier", "lane_change_not_performed", "steer_not_performed", "near_duplicate_plan"},
    "ball_force": {"target_mostly_out_of_view", "target_falls", "target_never_moves", "near_duplicate_plan",
                   "target_clipped_by_frame"},
    "beam_domino": {"no_causal_chain_in_unity", "target_never_moves", "falls_while_commanded", "near_duplicate_plan"},
    "liquid_container": {"bottle_tilts", "bottle_hovers", "near_duplicate_plan"},
    "character": {"character_move_not_performed", "target_falls", "near_duplicate_plan"},
}


def _stale_render(row: dict[str, Any]) -> bool:
    """True when the render on disk was made from different controls than the trajectory's action.npz."""
    render = ROOT / "runs/synthetic_unity" / row["path"].removeprefix("synthetic/")
    sched, cap = render / "schedule.json", render / "capture.json"
    if not (sched.is_file() and cap.is_file() and (render / "reference.mp4").is_file()):
        return True
    if json.loads(cap.read_text(encoding="utf-8")).get("status") != "ready":
        return True
    z = np.load(SUITE_ROOT / row["path"] / "action.npz", allow_pickle=False)
    d = json.loads(sched.read_text(encoding="utf-8"))
    vals = np.asarray(z["values"], dtype=np.float64).reshape(-1)
    return not (np.allclose(np.asarray(d["timestamps"]), z["timestamps_sec"]) and
                len(d["values"]) == len(vals) and np.allclose(np.asarray(d["values"]), vals))


def screen_flags(item: Item) -> list[dict[str, str]]:
    flags = [f for f in gt_checks(item)["flags"] if f["code"] in SCREEN_CODES.get(item.family, ())]
    if item.family == "beam_domino":
        from tools.dataset.build_synthetic_reference_inputs import _segments

        rec = _segments(item.root / "reference", item.item_id)
        if rec.get("trajectory_shape") is None:
            flags.append({"code": "no_causal_chain_in_unity", "severity": "major",
                          "detail": f"no causal chain in the Unity log ({rec.get('reason')})"})
    return flags


def near_duplicate_plans(family: str, threshold: float = 0.25) -> dict[str, dict[str, str]]:
    """Plans whose control signals differ by less than ``threshold`` (the generator test's
    criterion: of the family RMS, or for impulse families of the pair's own energy); flag the
    later slot."""
    sys.path.insert(0, str(ROOT))
    from tools.dataset import generate_synthetic_trajectories as gen
    fam = next(f for f in gen.families() if f["family"] == family)
    rows = [r for r in json.loads((SUITE_ROOT / "synthetic/manifest.json").read_text())["trajectories"]
            if r["family"] == family]
    sig = np.stack([gen.control_signal(fam, r["action_steps"], r["conditioning_case"]) for r in rows]).astype(float)
    if fam.get("signal_hold_sec"):
        energy = (sig ** 2).mean(-1)
        dist = np.sqrt(((sig[:, None] - sig[None]) ** 2).mean(-1) / ((energy[:, None] + energy[None]) / 2))
    else:   # tests/dataset/test_synthetic_trajectories.py's family-wide criterion
        dist = np.sqrt(((sig[:, None] - sig[None]) ** 2).mean(-1)) / np.sqrt((sig ** 2).mean())
    out = {}
    for j in range(len(rows)):
        for i in range(j):
            if dist[i, j] < threshold:
                out[rows[j]["synthetic_id"]] = {"code": "near_duplicate_plan", "severity": "major",
                                                "detail": f"actions nearly identical to {rows[i]['pattern']} ({dist[i, j]:.2f})"}
                break
    return out


def duplicate_tracks(items: list[Item]) -> dict[str, list[dict[str, str]]]:
    """cue_ball: references whose tracked (yellow) ball follows the same path as an earlier
    trajectory's -- the later strikes never reach it -- score the same thing; flag the later slot."""
    out: dict[str, list[dict[str, str]]] = {}
    seen: list[tuple[str, np.ndarray]] = []
    for i in sorted(items, key=lambda x: x.item_id):
        label = [s.strip() for s in str(i.module_params.get("kinematics_mechanics", {}).get("kin_text_prompt", "")).split(".") if s.strip()][0]
        path = _objects(i.artifacts).get(label)
        if path is None:
            continue
        for other, p in seen:
            if len(p) == len(path) and np.abs(p - path).max() < GT_DUPLICATE_TRACK_M:
                out.setdefault(i.item_id, []).append({"code": "duplicate_tracked_path", "severity": "major",
                                                      "detail": f"{label} follows the same path as in {other.rsplit('/', 1)[1]}"})
                break
        else:
            seen.append((i.item_id, path))
    return out


def screen(family: str, rounds: int, dry_run: bool) -> int:
    import subprocess

    py = sys.executable
    screening_path = SUITE_ROOT / "synthetic/screening.json"
    for rnd in range(1, rounds + 1):
        subprocess.run([py, "tools/dataset/generate_synthetic_trajectories.py"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        rows = [r for r in json.loads((SUITE_ROOT / "synthetic/manifest.json").read_text())["trajectories"]
                if r["family"] == family]
        stale = [r for r in rows if _stale_render(r)]
        print(f"[screen {family} round {rnd}] {len(rows)} trajectories, {len(stale)} to render", flush=True)
        if stale and not dry_run:
            args = [py, "tools/dataset/render_synthetic_references.py", "run", "--family", family, "--force"]
            for r in stale:
                args += ["--pattern", r["pattern"]]
            subprocess.run(args, cwd=ROOT, check=True)
        if stale:
            subprocess.run([py, "tools/dataset/render_synthetic_references.py", "publish"], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL)
            subprocess.run([py, "tools/dataset/generate_synthetic_trajectories.py"], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL)
        items = [i for i in load_items("synthetic") if i.family == family]
        doc = json.loads(screening_path.read_text()) if screening_path.is_file() else {
            "schema_version": 1,
            "note": ("Plans whose Unity reference failed the dataset audit (tools/dataset/audit_dataset.py screen); "
                     "tools/dataset/generate_synthetic_trajectories.py swaps each for the next-best plan of its stratum."),
            "rejected": {}}
        known = doc["rejected"].setdefault(family, [])
        new = []
        dup = duplicate_tracks(items) if family == "cue_ball" else {}
        for sid, flag in near_duplicate_plans(family).items():
            dup.setdefault(sid, []).append(flag)
        for i in items:
            flags = screen_flags(i) + dup.get(i.item_id, [])
            if flags:
                meta = json.loads((i.root / "meta.json").read_text(encoding="utf-8"))
                new.append({"synthetic_id": i.item_id, "conditioning_case": i.conditioning_case,
                            "plan": meta["action_steps"], "action_description": meta["action_description"],
                            "reason": "; ".join(f"{f['code']}: {f['detail']}" for f in flags)})
        print(f"[screen {family} round {rnd}] {len(new)} failing: {[n['synthetic_id'].rsplit('/', 1)[1] for n in new]}",
              flush=True)
        if not new:
            return 0
        if dry_run:
            return 1
        known += new
        screening_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[screen {family}] still failing after {rounds} rounds", flush=True)
    return 1


def _client(key_file: str, base_url: str):
    from openai import OpenAI

    key = Path(os.path.expanduser(key_file)).read_text(encoding="utf-8").strip()
    return OpenAI(base_url=base_url, api_key=key, timeout=180.0, max_retries=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "gt", "regt", "summary", "list", "screen"))
    ap.add_argument("--family", help="screen: the synthetic family to screen")
    ap.add_argument("--rounds", type=int, default=6, help="screen: max generate/render/check rounds")
    ap.add_argument("--dry-run", action="store_true", help="screen: report failures, change nothing")
    ap.add_argument("--split", choices=("all", "core", "synthetic"), default="all")
    ap.add_argument("--only", help="regex on item id")
    ap.add_argument("--round", default="r1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--key-file", default="~/.credentials/nv_vlm_api_key")
    ap.add_argument("--skip-vlm", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args(argv)
    out = OUT_ROOT / args.round
    if args.command == "summary":
        print(json.dumps(summarize(out, args.publish), indent=1))
        return 0
    if args.command == "screen":
        return screen(args.family, args.rounds, args.dry_run)
    items = load_items(args.split)
    if args.only:
        rx = re.compile(args.only)
        items = [i for i in items if rx.search(i.item_id)]
    if args.command == "list":
        for i in items:
            print(i.item_id, i.dimensions, i.not_applicable or "")
        print(len(items))
        return 0
    if args.command == "gt":
        for i in items:
            g = gt_checks(i)
            st = {k: v for k, v in g["stats"].items() if k not in ("video", "frozen")}
            print(i.item_id, json.dumps(g["flags"]), json.dumps(st, default=_np_default)[:600])
        return 0
    if args.command == "regt":
        # recompute the GT half (and the decision) of existing item records, keeping their VLM verdicts
        for i in items:
            path = out / "items" / f"{safe_name(i.item_id)}.json"
            if not path.is_file():
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            rec["gt"] = gt_checks(i)
            rec["decision"] = decide(rec["gt"], rec.get("vlm"))
            path.write_text(json.dumps(rec, indent=1, default=_np_default) + "\n", encoding="utf-8")
        print(json.dumps(summarize(out, args.publish), indent=1)[:3000])
        return 0
    client = None if args.skip_vlm else _client(args.key_file, args.base_url)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(audit_item, i, out, client, args.model, args.skip_vlm, args.force): i for i in items}
        for f in cf.as_completed(futs):
            i = futs[f]
            done += 1
            try:
                r = f.result()
                print(f"[{done}/{len(items)}] {r['decision']['status']:7s} {i.item_id}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[{done}/{len(items)}] ERROR {i.item_id}: {type(exc).__name__}: {exc}", flush=True)
    print(json.dumps(summarize(out, args.publish), indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
