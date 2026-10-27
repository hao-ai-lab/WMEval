#!/usr/bin/env python3
"""Ground truth for BallRoller/ctrl's kinematics cell: the controlled white ball.

Until 2026-10-21 the cell tracked only the static dome and cube (the ball's ground truth
was an open provenance question), so it measured hallucinated obstacle motion rather than
kinematics. This tool builds a trajectory for the controlled white ball that is consistent
with the core reference video (which is not changed):

1. Camera: the fixed Unity camera recovered for this case by render-and-compare
   (``render_synthetic_references.BALLROLLER_CAMERA``; a Unity render from it matches the
   core frame 0 at 25.3 dB) and its intrinsics, exported by the core-trajectory replay
   (``render_synthetic_references.py selfcheck --family ball_force``).
2. Why not the replay's own ball: replaying the core key script in the committed scene
   reproduces the approach (within ~10 px for the first second at 0.8x the scene's force
   gain) but not the contact -- the core recording's rack differs (untextured white balls,
   different response), so after contact the replayed ball is 100-400 px away from the
   recorded one. Its Unity log would contradict the reference video.
3. So the white ball is tracked in the core reference video itself (bright, unsaturated
   blob; nearest-blob association from its frame-0 position; checked frame by frame on
   overlays, ``--overlay``; 5-frame moving average), and its image centre is lifted through the Unity camera onto
   the horizontal plane at the ball's centre height (the replay's resting height, 0.476 m).
4. Plane calibration for metric FDE/DTW: static floor features of the core frame 0
   (``tools/run_unity_table_scenes._export_planar_calibration``, plane y = 0, the ball's
   start excluded), as for the Table 4 cases; candidates are tracked at the ball's
   bottom-centre.

Writes ``cases/BallRoller/ctrl/reference/artifacts/kin_*`` (the EOC and LPIPS artifacts are
untouched); re-author the case afterwards with ``tools/build_lite_case.py``.

    python tools/build_ballroller_kinematics_gt.py [--overlay out.jpg]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CASE = ROOT / "eval_sets/wmeval_lite_v1/cases/BallRoller/ctrl"
REPLAY = ROOT / "runs/synthetic_unity/_corecheck/ball_force/BallRoller__ctrl/artifacts"
BALL_CENTRE_Y = 0.476   # the replayed white ball's resting centre height (m)
FLOOR_Y = 0.0
START_UV = (731.0, 404.0)   # the white ball in the core frame 0
MAX_STEP_PX = 80.0


def frames(video: Path) -> list[np.ndarray]:
    cap, out = cv2.VideoCapture(str(video)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def track_white_ball(video_frames: list[np.ndarray]) -> np.ndarray:
    prev, out = np.asarray(START_UV), []
    for f in video_frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 2] > 200) & (hsv[..., 1] < 50)).astype(np.uint8)
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
        cands = [cent[k] for k in range(1, n) if stats[k, 4] >= 40]
        if cands:
            d = [float(np.hypot(*(c - prev))) for c in cands]
            j = int(np.argmin(d))
            if d[j] < MAX_STEP_PX:
                prev = np.asarray(cands[j])
        out.append(prev.copy())
    return np.asarray(out)


def lift(uv: np.ndarray, c2w: np.ndarray, k: np.ndarray, plane_y: float) -> np.ndarray:
    fx, fy, cx, cy = k
    ray = c2w[:3, :3] @ np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0])
    o = c2w[:3, 3]
    return o + (plane_y - o[1]) / ray[1] * ray


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlay", type=Path, help="write a contact sheet of the track for checking")
    args = ap.parse_args()
    from run_unity_table_scenes import _export_planar_calibration

    art = CASE / "reference/artifacts"
    vid = frames(CASE / "reference/reference.mp4")
    poses = np.load(REPLAY / "poses.npy")[: len(vid)]
    intr = np.load(REPLAY / "intrinsics.npy")[: len(vid)]
    if len(poses) != len(vid):
        raise SystemExit(f"replay has {len(poses)} frames, the core video {len(vid)}")
    raw_uv = track_white_ball(vid)
    # 5-frame centred moving average: pixel jitter becomes centimetres per frame once lifted
    # (the camera looks along the arena at a shallow angle)
    pad = np.pad(raw_uv, ((2, 2), (0, 0)), mode="edge")
    uv = np.stack([pad[i:i + len(raw_uv)] for i in range(5)]).mean(axis=0)
    world = np.asarray([lift(uv[i], poses[i], intr[i], BALL_CENTRE_Y) for i in range(len(vid))])
    np.save(art / "kin_poses.npy", poses, allow_pickle=False)
    np.save(art / "kin_intrinsics.npy", intr, allow_pickle=False)
    (art / "kin_object_transforms.json").write_text(json.dumps({
        "schema_version": 1,
        "source": ("white ball tracked in reference/reference.mp4 and lifted through the recovered Unity camera "
                   "onto y = 0.476 m (tools/build_ballroller_kinematics_gt.py)"),
        "objects": {"white ball": {"frames": list(range(len(world))),
                                   "positions_world": [[round(float(v), 5) for v in p] for p in world],
                                   "pixel_uv": [[round(float(v), 2) for v in p] for p in uv]}},
    }, indent=1) + "\n", encoding="utf-8")
    (art / "kin_object_masses.json").write_text(json.dumps({"white ball": 1.0}, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        f0 = Path(tmp) / "frame0.png"
        cv2.imwrite(str(f0), vid[0])
        rows = [{"balls": {"white ball": world[0].tolist()}, "motion_plane_y": FLOOR_Y}]
        cal, overlay = _export_planar_calibration(Path(tmp), f0, rows, poses, intr, "billiard")
        doc = json.loads(cal.read_text())
        doc["notes"].append("BallRoller/ctrl: features of the core reference frame 0 on the floor plane (y = 0) "
                            "through the recovered Unity camera.")
        (art / "kin_plane_calibration.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"{len(world)} frames; plane anchors {len(doc['anchors'])} ({doc['status']}); "
          f"path {np.sum(np.linalg.norm(np.diff(world[:, [0, 2]], axis=0), axis=1)):.2f} m")
    if args.overlay:
        idx = np.linspace(0, len(vid) - 1, 16).astype(int)
        tiles = []
        for i in idx:
            f = vid[i].copy()
            cv2.circle(f, tuple(int(v) for v in uv[i]), 16, (0, 0, 255), 2)
            crop = cv2.resize(f[240:620, 160:1120], (480, 190))
            cv2.putText(crop, str(i), (5, 22), 0, 0.7, (0, 0, 255), 2)
            tiles.append(crop)
        cv2.imwrite(str(args.overlay), np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
