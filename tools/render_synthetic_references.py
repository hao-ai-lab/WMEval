#!/usr/bin/env python3
"""Render Unity reference videos for the synthetic ``test_full`` trajectories.

Every synthetic trajectory (``eval_sets/wmeval_lite_v1/synthetic/manifest.json``)
is replayed in the Unity scene of its conditioning core case: the trajectory's
compiled ``action.npz`` is turned into a ``schedule.json`` (zero-order-held
control steps, plus precomputed per-frame camera yaw/poses for camera
families) and passed to a Unity Player or Editor batch method via
``-wmeval-schedule``. The scene-side replay hooks live on the local
``wmeval/synthetic-replay`` branch of WorldModelBenchScenePrivate.

Outputs, per trajectory, under ``runs/synthetic_unity/<synthetic_id>/``::

    reference.mp4        h264 / yuv420p, core-case fps and resolution
    ref.png              rendered frame 0 (compare with the core first frame)
    artifacts/           Unity ground truth the family's dimensions consume
    schedule.json        exactly what Unity replayed
    capture.json         command, build hash, fps, frames, duration, status
    unity.log            Player / Editor log

Usage::

    python tools/render_synthetic_references.py list
    python tools/render_synthetic_references.py run --family cue_ball --limit 1
    python tools/render_synthetic_references.py run --pilot          # 1 per family
    python tools/render_synthetic_references.py run --detach         # all, nohup'd
    python tools/render_synthetic_references.py status
    python tools/render_synthetic_references.py publish      # copy into the dataset tree + references.json

``run`` is resumable: a trajectory whose ``capture.json`` says ``ready`` (and
whose ``reference.mp4`` exists) is skipped unless ``--force``. ``--detach``
re-executes the same command in a new session (``setsid``-style, stdin closed,
output appended to ``runs/synthetic_unity/run.log``) and writes the PID to
``runs/synthetic_unity/run.pid``.

References are never synthesized outside Unity: a family whose replay backend
is unavailable is recorded as ``blocked`` with the reason.

Replay backends (Unity side on WorldModelBenchScenePrivate ``wmeval/synthetic-replay``;
``WMEvalSchedule.cs`` is the shared schedule reader in every project):

* camera_yaw / WanderingGarden: ``GardenGTCapture`` per-frame yaw (``frame_yaw_deg``).
* camera_yaw / BilliardGame: ``BilliardCaptureBootstrap -wmeval-camera-mode schedule``.
* camera_yaw / MirrorRollingBall, ChessBoard (prompt views): ``WMEvalReplayCapture``
  ``camera_poses``; RacingCar prompt views: ``RacingCaptureBootstrap`` with frame poses.
  Poses = the core frame-0 camera yawed in place (first person) or about the
  character (third person) by the integrated ``yaw_rate_deg_s``.
* camera_orbit: Editor batch method ``WMEvalOrbitReplay.Run`` (the core case's look-at
  orbit geometry, per-frame orbit phase) on the scene file of revision 84a06a4, with the
  native camera rig, sun and post-processing of the paper's SunRisesAndSets/lighting frame.
* ego_vehicle: ``RacingCaptureBootstrap`` -> ``CarController.SetInput`` per frame.
* cue_ball: ``BilliardCaptureBootstrap -wmeval-control-mode schedule`` (Shot() impulses).
* ball_force: ``WMEvalReplayCapture rigidbody_force`` (AddForce with the scene's gain).
* beam_domino: ``BeamDominoAgent.Heuristic`` reads the schedule (HeuristicOnly).
* character: ``ChessCaptureBootstrap`` -> ``StarterAssetsInputs``.
* liquid_container: ``WMEvalReplayCapture bottle_input`` (needs the scene's LFS assets checked out).

Scene variants: a spec may build from an older scene revision (``source_rev``, a sparse
worktree of that commit plus the replay scripts) or from the current project with only
the LFS assets that existed at a given commit (``lfs_rev``), so each render starts from
what its core case was recorded in. Both are prepared under ``runs/synthetic_unity/projects``.

``selfcheck`` replays each physics family's core trajectory through the same backend
and compares with the core reference video (``runs/synthetic_unity/_corecheck``).

``core --case SunRisesAndSets/lighting`` renders a core case's own trajectory through its
family backend (``runs/synthetic_unity/_core``); ``tools/build_lighting_case.py`` turns
that render into the case's reference video and Unity ground truth.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUITE = ROOT / "eval_sets/wmeval_lite_v1"
MANIFEST = SUITE / "synthetic/manifest.json"
OUT = ROOT / "runs/synthetic_unity"
UNITY_REPO = (ROOT.parent / "WorldModelBenchScenePrivate").resolve()
BUILDS = OUT / "builds"
EDITOR_ROOT = Path("/Applications/Unity/Hub/Editor")
DEFAULT_EDITOR = "6000.3.7f1"
JOB_TIMEOUT_SEC = 900
PROJECTS = OUT / "projects"     # variant Unity projects (see SceneSpec.source_rev / lfs_rev)
# The scene assets live in GitHub's LFS store (a clone may point lfs.url elsewhere).
LFS_REMOTE = "https://github.com/kunqianut/WorldModelBenchScenePrivate.git"
# Replay/build scripts copied into a variant checkout of an older scene revision.
OVERLAY = ("Assets/WMEval", "Assets/WMEval.meta", "Assets/Editor/BuildScript.cs", "Assets/Editor/BuildScript.cs.meta")


# --------------------------------------------------------------------------- utils
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_manifest() -> list[dict[str, Any]]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["trajectories"]


def case_dir(case_id: str) -> Path:
    return SUITE / "cases" / case_id


def editor(version: str = DEFAULT_EDITOR) -> Path:
    return EDITOR_ROOT / version / "Unity.app/Contents/MacOS/Unity"


def app_binary(app: Path) -> Path:
    macos = app / "Contents/MacOS"
    preferred = macos / app.stem
    if preferred.is_file():
        return preferred
    found = [p for p in macos.iterdir() if p.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"cannot identify player binary in {app}")
    return found[0]


def build_hash(app: Path) -> str:
    """Hash of the player's managed code + binary (identifies the replay build)."""
    h = hashlib.sha256()
    managed = app / "Contents/Resources/Data/Managed"
    paths = [app_binary(app)]
    if managed.is_dir():
        paths += sorted(p for p in managed.glob("Assembly-CSharp*.dll"))
    for p in paths:
        h.update(p.name.encode())
        h.update(sha256(p).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- schedules
def load_action(traj_dir: Path) -> dict[str, Any]:
    z = np.load(traj_dir / "action.npz", allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["channel_names"] = [str(c) for c in out["channel_names"].tolist()]
    return out


def held(action: dict[str, Any], channel: str, times: np.ndarray, end_sec: float) -> np.ndarray:
    ts = np.asarray(action["timestamps_sec"], dtype=np.float64)
    ci = action["channel_names"].index(channel)
    idx = np.clip(np.searchsorted(ts, times + 1e-9, side="right") - 1, 0, len(ts) - 1)
    vals = np.asarray(action["values"], dtype=np.float64)[idx, ci]
    vals[times >= end_sec - 1e-9] = 0.0
    return vals


def integrated_yaw(action: dict[str, Any], fps: float, frames: int, end_sec: float) -> np.ndarray:
    """Yaw offset (deg) at frame k = integral of yaw_rate over [0, k/fps]."""
    ts = np.asarray(action["timestamps_sec"], dtype=np.float64)
    rate = np.asarray(action["values"], dtype=np.float64)[:, action["channel_names"].index("yaw_rate_deg_s")]
    edges = np.append(ts, end_sec)
    out = np.zeros(frames)
    for k in range(frames):
        t = min(k / fps, end_sec)
        dt = np.clip(t - edges[:-1], 0.0, np.diff(edges))
        out[k] = float(np.sum(dt * rate))
    return out


def schedule_doc(row: dict[str, Any], spec: "SceneSpec", action: dict[str, Any]) -> dict[str, Any]:
    duration = float(row["duration_sec"])
    frames = spec.frames_for(duration)
    doc: dict[str, Any] = {
        "schema_version": 1,
        "synthetic_id": row["synthetic_id"],
        "family": row["family"],
        "pattern": row["pattern"],
        "mode": spec.mode,
        "fps": spec.fps,
        "frames": frames,
        "width": spec.width,
        "height": spec.height,
        "duration_sec": duration,
        "control_hz": float(row["control_hz"]),
        "channels": action["channel_names"],
        "timestamps": [float(t) for t in action["timestamps_sec"]],
        "values": [float(v) for v in np.asarray(action["values"], dtype=np.float64).reshape(-1)],
        "vertical_fov_deg": 0.0,
        "param_names": [],
        "param_values": [],
    }
    if "yaw_rate_deg_s" in action["channel_names"]:
        doc["frame_yaw_deg"] = [float(v) for v in integrated_yaw(action, spec.fps, frames, duration)]
    if spec.prepare is not None:
        spec.prepare(doc, row, action)
    return doc


# --------------------------------------------------------------------------- scene specs
@dataclass
class SceneSpec:
    key: str                      # family/conditioning_case
    project: str                  # GameScenes/<Project>
    scene: str                    # canonical scene path inside the project
    product: str
    fps: int
    width: int
    height: int
    mode: str
    kind: str = "player"          # player | editor
    editor_version: str = DEFAULT_EDITOR
    runtime_args: list[str] = field(default_factory=list)
    editor_method: str = ""
    frames_override: Callable[[float], int] | None = None
    encode_frames: Callable[[int], int] | None = None
    prepare: Callable[[dict, dict, dict], None] | None = None
    export: Callable[[Path, Path, dict, "SceneSpec"], dict[str, Any]] | None = None
    post_encode: Callable[[Path, "SceneSpec", int], dict[str, Any]] | None = None
    first_frame_ref: str | None = None   # path relative to the core case dir
    blocked: str | None = None
    # Scene/asset variant, so the render starts from what the core case was recorded in:
    source_rev: str | None = None   # build from a sparse checkout of this Unity-repo commit
    lfs_rev: str | None = None      # build the current project with only the LFS assets present at this commit
    asset_note: str = ""            # human-readable asset provenance (published in references.json)

    def frames_for(self, duration: float) -> int:
        if self.frames_override is not None:
            return self.frames_override(duration)
        return int(round(duration * self.fps))

    @property
    def app(self) -> Path:
        return BUILDS / self.product / f"{self.product}.app"


SPECS: dict[str, SceneSpec] = {}   # filled by register_specs()


def spec_for(row: dict[str, Any]) -> SceneSpec:
    key = f"{row['family']}/{row['conditioning_case']}"
    if key not in SPECS:
        raise KeyError(f"no replay backend registered for {key}")
    return SPECS[key]


# --------------------------------------------------------------------------- jobs
def job_dir(row: dict[str, Any]) -> Path:
    sid = row["synthetic_id"]
    if sid.startswith("corecheck/"):
        return OUT / "_corecheck" / sid[len("corecheck/"):]
    if sid.startswith("core/"):
        return OUT / "_core" / sid[len("core/"):]
    assert sid.startswith("synthetic/")
    return OUT / sid[len("synthetic/"):]


SELFCHECK_FAMILIES = {  # family -> core case whose own trajectory is replayed through the same backend
    "ego_vehicle": "RacingCar/kinematics_third", "cue_ball": "BilliardGame/kinematics_first",
    "ball_force": "BallRoller/ctrl", "ball_force_mirror": "MirrorRollingBall/lpips_collision",
    "beam_domino": "BeamDomino/causal", "liquid_container": "LiquidContainer/fluid_rest",
    "character": "ChessBoard/walk_third",
    # the liquid profiles with motion (fluid_rest alone cannot validate the bottle physics)
    "liquid_container:step_hold": "LiquidContainer/fluid_step_hold",
    "liquid_container:impulse_decay": "LiquidContainer/fluid_impulse_decay",
    "liquid_container:chirp_strafe": "LiquidContainer/fluid_chirp_strafe",
}


def selfcheck_rows() -> list[dict[str, Any]]:
    import yaml

    rows = []
    for fam, case in SELFCHECK_FAMILIES.items():
        family = fam.split(":")[0].replace("_mirror", "")
        art = f"cases/{case}/reference/artifacts"
        traj = yaml.safe_load((SUITE / art / "trajectory.yaml").read_text(encoding="utf-8"))
        rows.append({"synthetic_id": f"corecheck/{family}/{case.replace('/', '__')}", "family": family,
                     "pattern": "core_trajectory", "conditioning_case": case, "path": art,
                     "trajectory_id": traj["trajectory_id"], "duration_sec": float(traj["duration_sec"]),
                     "control_hz": float(traj["control_hz"])})
    return rows


CORE_FAMILIES = {"SunRisesAndSets/lighting": "camera_orbit"}  # core case -> backend rendering its own trajectory


def core_rows(cases: list[str] | None = None) -> list[dict[str, Any]]:
    """The core cases' own trajectories as render rows (reference video + ground truth of the case)."""
    import yaml

    rows = []
    for case, family in CORE_FAMILIES.items():
        if cases and case not in cases:
            continue
        art = f"cases/{case}/reference/artifacts"
        traj = yaml.safe_load((SUITE / art / "trajectory.yaml").read_text(encoding="utf-8"))
        rows.append({"synthetic_id": f"core/{family}/{case.replace('/', '__')}", "family": family,
                     "pattern": "core_trajectory", "conditioning_case": case, "path": art,
                     "trajectory_id": traj["trajectory_id"], "duration_sec": float(traj["duration_sec"]),
                     "control_hz": float(traj["control_hz"])})
    return rows


def video_psnr(a: Path, b: Path, max_frames: int = 100000) -> list[float]:
    import cv2

    ca, cb = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    out = []
    while len(out) < max_frames:
        ok1, x = ca.read()
        ok2, y = cb.read()
        if not (ok1 and ok2):
            break
        if x.shape != y.shape:
            y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
        mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
        out.append(99.0 if mse < 1e-10 else 10 * math.log10(255.0 ** 2 / mse))
    return out


def is_done(row: dict[str, Any]) -> bool:
    d = job_dir(row)
    cap = d / "capture.json"
    if not cap.is_file() or not (d / "reference.mp4").is_file():
        return False
    try:
        return json.loads(cap.read_text(encoding="utf-8")).get("status") == "ready"
    except json.JSONDecodeError:
        return False


def select(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = rows
    if args.family:
        out = [r for r in out if r["family"] in set(args.family)]
    if args.case:
        out = [r for r in out if r["conditioning_case"] in set(args.case)]
    if args.pattern:
        out = [r for r in out if r["pattern"] in set(args.pattern)]
    if getattr(args, "pilot", False):
        seen, pilot = set(), []
        for r in out:
            if r["family"] not in seen:
                seen.add(r["family"])
                pilot.append(r)
        out = pilot
    if args.limit:
        if args.family and len(args.family) > 1 or not args.family:
            # limit per family keeps pilots balanced when several families are selected
            per: dict[str, int] = {}
            limited = []
            for r in out:
                per[r["family"]] = per.get(r["family"], 0) + 1
                if per[r["family"]] <= args.limit:
                    limited.append(r)
            out = limited
        else:
            out = out[: args.limit]
    return out


def encode(frames_dir: Path, fps: int, count: int, dest: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    pattern = sorted(frames_dir.glob("frame_*.png"))
    if not pattern:
        raise RuntimeError(f"no frames in {frames_dir}")
    digits = len(pattern[0].stem.split("_")[-1])
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps), "-start_number", "0",
         "-i", str(frames_dir / f"frame_%0{digits}d.png"), "-frames:v", str(count), "-an",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
        check=True,
    )


def probe(video: Path) -> dict[str, Any]:
    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,r_frame_rate,nb_read_frames,codec_name,pix_fmt", "-of", "json", str(video)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(raw)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": s["width"], "height": s["height"], "fps": float(num) / float(den),
            "frames": int(s["nb_read_frames"]), "codec": s["codec_name"], "pix_fmt": s["pix_fmt"]}


def first_frame_psnr(frame0: Path, row: dict[str, Any], spec: SceneSpec) -> dict[str, Any] | None:
    """PSNR of our frame 0 against the core case's first frame (ref.png or video frame 0)."""
    import cv2

    ours = cv2.imread(str(frame0), cv2.IMREAD_COLOR)
    core = case_dir(row["conditioning_case"])
    ref_path = core / (spec.first_frame_ref or "reference/ref.png")
    if ref_path.suffix == ".png" and ref_path.is_file():
        ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        source = str(ref_path.relative_to(ROOT))
    else:
        vids = sorted((core / "reference").glob("*.mp4"))
        if not vids:
            return None
        cap = cv2.VideoCapture(str(vids[0]))
        ok, ref = cap.read()
        cap.release()
        if not ok:
            return None
        source = f"{vids[0].relative_to(ROOT)}#frame0"
    if ours is None or ref is None:
        return None
    if ref.shape != ours.shape:
        ref = cv2.resize(ref, (ours.shape[1], ours.shape[0]), interpolation=cv2.INTER_AREA)
    mse = float(np.mean((ours.astype(np.float64) - ref.astype(np.float64)) ** 2))
    psnr = 99.0 if mse <= 1e-10 else 10 * math.log10(255.0 ** 2 / mse)
    return {"reference": source, "psnr_db": round(psnr, 3), "mae": round(float(np.mean(np.abs(ours.astype(float) - ref.astype(float)))), 3)}


def run_one(row: dict[str, Any], force: bool = False) -> dict[str, Any]:
    spec = spec_for(row)
    d = job_dir(row)
    if not force and is_done(row):
        return {"synthetic_id": row["synthetic_id"], "status": "skipped_done"}
    if spec.blocked:
        d.mkdir(parents=True, exist_ok=True)
        cap = {"schema_version": 1, "synthetic_id": row["synthetic_id"], "trajectory_id": row["trajectory_id"],
               "family": row["family"], "pattern": row["pattern"], "conditioning_case": row["conditioning_case"],
               "status": "blocked", "reason": spec.blocked, "updated": now()}
        write_json(d / "capture.json", cap)
        return cap
    if d.exists():
        for child in ("frames", "raw", "artifacts"):
            shutil.rmtree(d / child, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    traj_dir = SUITE / row["path"]
    action = load_action(traj_dir)
    sched = schedule_doc(row, spec, action)
    write_json(d / "schedule.json", sched)
    raw = d / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    unity_log = d / "unity.log"
    started = time.time()
    if spec.kind == "player":
        if not spec.app.is_dir():
            raise FileNotFoundError(f"replay build missing, run `build` first: {spec.app}")
        binary = app_binary(spec.app)
        command = [str(binary), "-batchmode", "-logFile", str(unity_log), "-outdir", str(raw), "-outDir", str(raw),
                   "-wmeval-schedule", str(d / "schedule.json"),
                   "-screen-width", str(spec.width), "-screen-height", str(spec.height), *spec.runtime_args]
        bhash = build_hash(spec.app)
    else:
        project = UNITY_REPO / spec.project
        command = [str(editor(spec.editor_version)), "-batchmode", "-projectPath", str(project),
                   "-executeMethod", spec.editor_method, "-logFile", str(unity_log),
                   "-outdir", str(raw), "-wmeval-schedule", str(d / "schedule.json"), *spec.runtime_args]
        bhash = "editor:" + spec.editor_version + ":" + git_rev(project)
    cap: dict[str, Any] = {
        "schema_version": 1, "synthetic_id": row["synthetic_id"], "trajectory_id": row["trajectory_id"],
        "family": row["family"], "pattern": row["pattern"], "conditioning_case": row["conditioning_case"],
        "scene_project": spec.project, "scene": spec.scene, "replay_mode": spec.mode,
        "command": command, "build_hash": bhash, "fps": spec.fps, "width": spec.width, "height": spec.height,
        "trajectory_duration_sec": float(row["duration_sec"]), "status": "running", "started": now(),
    }
    write_json(d / "capture.json", cap)
    try:
        proc = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.STDOUT, timeout=JOB_TIMEOUT_SEC)
        cap["unity_exit_code"] = proc.returncode
        frames_dir = raw / "frames"
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        expected = sched["frames"]
        if len(frame_files) < expected:
            raise RuntimeError(f"Unity wrote {len(frame_files)} frames, expected {expected} (exit {proc.returncode})")
        n_video = spec.encode_frames(expected) if spec.encode_frames else expected
        encode(frames_dir, spec.fps, n_video, d / "reference.mp4")
        shutil.copyfile(frame_files[0], d / "ref.png")
        artifacts_dir = d / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        extra = spec.export(raw, artifacts_dir, row, spec) if spec.export else {}
        post = spec.post_encode(d, spec, n_video) if spec.post_encode else {}
        extra.update(post)
        info = probe(d / "reference.mp4")
        if not post and (info["frames"] != n_video or (info["width"], info["height"]) != (spec.width, spec.height)):
            raise RuntimeError(f"encoded video mismatch: {info}")
        import cv2
        img = cv2.imread(str(d / "ref.png"))
        if img is None or float(img.std()) < 2.0:
            raise RuntimeError("flat or unreadable frame 0 (graphics device disabled?)")
        cap.update({
            "status": "ready", "frame_count": info["frames"], "rendered_frames": len(frame_files),
            "duration_sec": round(info["frames"] / info["fps"], 6), "video": info,
            "first_frame_check": first_frame_psnr(d / "ref.png", row, spec),
            "artifacts": {p.name: sha256(p) for p in sorted(artifacts_dir.iterdir()) if p.is_file()},
            "reference_sha256": sha256(d / "reference.mp4"), **extra,
        })
        # PNG frames are large; keep only the encoded video + frame 0 + GT.
        shutil.rmtree(frames_dir, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - job-level failure is recorded, run continues
        cap.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=4)})
    cap["elapsed_sec"] = round(time.time() - started, 1)
    cap["finished"] = now()
    write_json(d / "capture.json", cap)
    return cap


def git_rev(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip() + "+wt"
    except Exception:  # noqa: BLE001
        return "unknown"


REFERENCES = SUITE / "synthetic/references.json"
PUBLISH_FILES = ("reference.mp4", "reference_native.mp4", "ref.png", "capture.json", "schedule.json")
WEAK_FIRST_FRAME_DB = 20.0  # below this, frame 0 visibly differs from the core conditioning frame


def _scrub(text: str) -> str:
    """Replace machine-local absolute paths in published JSON."""
    for real, alias in ((str(UNITY_REPO), "$UNITY_REPO"), (str(ROOT), "$WMEVAL"), (str(Path.home()), "~")):
        text = text.replace(real, alias)
    return text


def publish(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Copy finished references into the dataset tree and write synthetic/references.json.

    Each trajectory gets ``reference/`` next to its trajectory files (video, frame 0, Unity
    artifacts, capture record, replayed schedule); references.json lists every file's sha256
    plus the render provenance, and is what the generator and the sync tool read.
    """
    entries: dict[str, Any] = {}
    for r in rows:
        if not is_done(r):
            continue
        d, spec = job_dir(r), spec_for(r)
        cap = json.loads((d / "capture.json").read_text(encoding="utf-8"))
        dest = SUITE / r["path"] / "reference"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        for name in PUBLISH_FILES:
            if (d / name).is_file():
                shutil.copy2(d / name, dest / name)
        if (d / "artifacts").is_dir():
            shutil.copytree(d / "artifacts", dest / "artifacts")
        for j in dest.rglob("*.json"):
            j.write_text(_scrub(j.read_text(encoding="utf-8")), encoding="utf-8")
        psnr = (cap.get("first_frame_check") or {}).get("psnr_db")
        entries[r["synthetic_id"]] = {
            "video": "reference/reference.mp4",
            "files": {str(f.relative_to(dest.parent)): sha256(f) for f in sorted(dest.rglob("*")) if f.is_file()},
            "fps": cap.get("fps"), "width": cap.get("width"), "height": cap.get("height"),
            "frames": cap.get("frame_count"), "duration_sec": cap.get("duration_sec"),
            "first_frame_psnr_db": psnr,
            "quality": "weak_first_frame" if psnr is not None and psnr < WEAK_FIRST_FRAME_DB else "ok",
            "unity": {"scene_project": spec.project, "scene": spec.scene, "replay_mode": spec.mode,
                      "source_rev": spec.source_rev, "lfs_rev": spec.lfs_rev,
                      "asset_note": spec.asset_note or f"{spec.project} at the Unity revision above, all its LFS assets present",
                      "build_hash": cap.get("build_hash")},
        }
    doc = {"schema_version": 1, "unity_repo": "kunqianut/WorldModelBenchScenePrivate",
           "unity_repo_rev": git_rev(UNITY_REPO), "weak_first_frame_threshold_db": WEAK_FIRST_FRAME_DB,
           "count": len(entries), "references": dict(sorted(entries.items()))}
    write_json(REFERENCES, doc)
    return doc


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    items = []
    for r in rows:
        cap_path = job_dir(r) / "capture.json"
        status = "pending"
        extra: dict[str, Any] = {}
        if cap_path.is_file():
            try:
                cap = json.loads(cap_path.read_text(encoding="utf-8"))
                status = cap.get("status", "unknown")
                extra = {k: cap.get(k) for k in ("error", "reason", "frame_count", "elapsed_sec", "first_frame_check") if cap.get(k) is not None}
            except json.JSONDecodeError:
                status = "corrupt"
        by_status[status] = by_status.get(status, 0) + 1
        fam = by_family.setdefault(r["family"], {})
        fam[status] = fam.get(status, 0) + 1
        items.append({"synthetic_id": r["synthetic_id"], "status": status, **extra})
    return {"schema_version": 1, "updated": now(), "total": len(rows), "by_status": by_status,
            "by_family": by_family, "trajectories": items}


# --------------------------------------------------------------------------- builds
def _git(*args: str, cwd: Path = UNITY_REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _lfs(*args: str, cwd: Path) -> None:
    subprocess.run(["git", "-c", f"lfs.url={LFS_REMOTE}/info/lfs", "-c", "lfs.sshtransfer=negotiate", "lfs", *args],
                   cwd=cwd, check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def _copy_overlay(spec: SceneSpec, project: Path) -> None:
    """(Re)copy the replay/build scripts from the main Unity working tree into a source_rev variant."""
    for rel in OVERLAY:
        src, dst = UNITY_REPO / spec.project / rel, project / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def project_dir(spec: SceneSpec) -> Path:
    """The Unity project a spec builds from, preparing a variant project when it declares one."""
    if not (spec.source_rev or spec.lfs_rev):
        return UNITY_REPO / spec.project
    root = PROJECTS / spec.product
    marker = root / ".wmeval_variant.json"
    want = {"project": spec.project, "source_rev": spec.source_rev, "lfs_rev": spec.lfs_rev}
    project = root / spec.project
    if marker.is_file() and json.loads(marker.read_text()) == want:
        if spec.source_rev:
            _copy_overlay(spec, project)   # replay scripts may have changed since the variant was prepared
        return project
    if root.exists():
        if spec.source_rev:
            subprocess.run(["git", "worktree", "remove", "--force", str(root)], cwd=UNITY_REPO, capture_output=True)
        shutil.rmtree(root, ignore_errors=True)
    if spec.source_rev:
        # An older scene revision: sparse worktree of that commit, its LFS assets, our replay scripts on top.
        _git("worktree", "add", "--no-checkout", "--detach", str(root), spec.source_rev)
        _git("sparse-checkout", "set", spec.project, cwd=root)
        _git("checkout", cwd=root)
        _lfs("fetch", LFS_REMOTE, "--include", f"{spec.project}/**", cwd=root)
        _lfs("checkout", spec.project, cwd=root)
        _copy_overlay(spec, project)
    else:
        # The current project, but only with the LFS assets that existed at lfs_rev (later ones
        # stay pointers, i.e. missing, as on the machine that recorded the core reference).
        src = UNITY_REPO / spec.project
        for sub in ("Assets", "Packages", "ProjectSettings"):
            shutil.copytree(src / sub, project / sub, symlinks=True)
        keep = {line.split()[0] for line in _git("lfs", "ls-files", "-l", spec.lfs_rev).splitlines() if line.strip()}
        for line in _git("lfs", "ls-files", "-l").splitlines():
            oid, _, path = line.split(" ", 2)
            path = path.lstrip("*- ").strip()
            if path.startswith(spec.project + "/") and oid not in keep:
                (root / path).write_text(_git("cat-file", "-p", f"HEAD:{path}"), encoding="utf-8")
    marker.write_text(json.dumps(want), encoding="utf-8")
    return project


def build(spec: SceneSpec) -> None:
    if spec.kind != "player":
        return
    project = project_dir(spec)
    spec.app.parent.mkdir(parents=True, exist_ok=True)
    blog = OUT / "logs" / f"build_{spec.product}.log"
    blog.parent.mkdir(parents=True, exist_ok=True)
    # Never -nographics: it disables Metal and the replayed Player renders flat gray.
    cmd = [str(editor(spec.editor_version)), "-batchmode", "-quit", "-projectPath", str(project),
           "-buildTarget", "OSXUniversal", "-executeMethod", "BuildScript.BuildMacOS",
           "-buildPath", str(spec.app.parent), "-productName", spec.product, "-scenes", spec.scene,
           "-logFile", str(blog)]
    log(f"building {spec.product}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if not spec.app.is_dir():
        raise RuntimeError(f"Unity reported success but {spec.app} is missing (see {blog})")
    log(f"built {spec.app} hash={build_hash(spec.app)[:12]}")


# --------------------------------------------------------------------------- CLI
def detach(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_log = OUT / "run.log"
    argv = [a for a in argv if a != "--detach"]
    cmd = [sys.executable, str(Path(__file__).resolve()), *argv]
    with run_log.open("ab") as fh:
        fh.write(f"\n===== detached launch {now()}: {' '.join(cmd)}\n".encode())
        proc = subprocess.Popen(["nohup", *cmd], stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), start_new_session=True, close_fds=True)
    (OUT / "run.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
    print(json.dumps({"pid": proc.pid, "log": str(run_log), "status": str(OUT / "status.json"),
                      "command": cmd}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("list", "build", "run", "status", "selfcheck", "publish", "core"))
    p.add_argument("--family", action="append")
    p.add_argument("--case", action="append", help="conditioning core case id")
    p.add_argument("--pattern", action="append")
    p.add_argument("--limit", type=int, default=0, help="max trajectories (per family when several)")
    p.add_argument("--pilot", action="store_true", help="first trajectory of each family")
    p.add_argument("--force", action="store_true", help="re-render completed trajectories")
    p.add_argument("--retry-failed", action="store_true", help="(default) failed jobs are retried on rerun")
    p.add_argument("--detach", action="store_true", help="run in the background (nohup, new session)")
    args = p.parse_args(argv)
    if args.detach:
        return detach(argv)
    rows = load_manifest()
    chosen = select(rows, args)
    if args.command == "list":
        for r in chosen:
            spec = SPECS.get(f"{r['family']}/{r['conditioning_case']}")
            state = "done" if is_done(r) else ("blocked" if spec and spec.blocked else "todo")
            print(f"{state:8s} {r['synthetic_id']}")
        print(f"{len(chosen)} trajectories")
        return 0
    if args.command == "publish":
        doc = publish(rows)
        quality: dict[str, int] = {}
        for e in doc["references"].values():
            quality[e["quality"]] = quality.get(e["quality"], 0) + 1
        print(json.dumps({"published": doc["count"], "of": len(rows), "quality": quality,
                          "references": str(REFERENCES.relative_to(ROOT))}, indent=2))
        return 0
    if args.command == "status":
        summary = summarize(rows)
        write_json(OUT / "status.json", summary)
        print(json.dumps({k: summary[k] for k in ("updated", "total", "by_status", "by_family")}, indent=2))
        return 0
    if args.command == "selfcheck":
        # Replay each physics family's *core* trajectory through the same backend and
        # compare with the core reference video (validates gains, timing and camera).
        report = []
        for r in selfcheck_rows():
            if args.family and r["family"] not in args.family:
                continue
            res = run_one(r, force=args.force)
            d = job_dir(r)
            core_vids = sorted((case_dir(r["conditioning_case"]) / "reference").glob("*.mp4"))
            curve = video_psnr(d / "reference.mp4", core_vids[0]) if (d / "reference.mp4").is_file() else []
            item = {"case": r["conditioning_case"], "status": res.get("status") or json.loads((d / "capture.json").read_text()).get("status"),
                    "frames_compared": len(curve), "psnr_frame0": round(curve[0], 2) if curve else None,
                    "psnr_mean": round(float(np.mean(curve)), 2) if curve else None,
                    "psnr_min": round(float(np.min(curve)), 2) if curve else None,
                    "psnr_by_second": [round(float(np.mean(curve[i:i + 30])), 1) for i in range(0, len(curve), 30)]}
            log(f"selfcheck {item}")
            report.append(item)
        write_json(OUT / "_corecheck" / "selfcheck.json", {"updated": now(), "items": report})
        return 0
    if args.command == "core":
        # Render core cases' own trajectories (their reference video + Unity ground truth).
        for r in core_rows(args.case):
            res = run_one(r, force=True)
            log(f"core {r['conditioning_case']} -> {res.get('status')} {res.get('error', '')} "
                f"psnr0={((res.get('first_frame_check') or {}).get('psnr_db'))} dir={job_dir(r)}")
        return 0
    if args.command == "build":
        done: set[str] = set()
        for r in chosen:
            spec = spec_for(r)
            if spec.product in done or spec.blocked or spec.kind != "player":
                continue
            done.add(spec.product)
            build(spec)
        return 0
    # run
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    log(f"run start pid={os.getpid()} selected={len(chosen)}")
    t0 = time.time()
    for i, r in enumerate(chosen, 1):
        if not args.force and is_done(r):
            continue
        log(f"[{i}/{len(chosen)}] {r['synthetic_id']}")
        try:
            res = run_one(r, force=args.force)
        except Exception as exc:  # noqa: BLE001
            res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            d = job_dir(r)
            d.mkdir(parents=True, exist_ok=True)
            write_json(d / "capture.json", {"schema_version": 1, "synthetic_id": r["synthetic_id"], **res,
                                            "updated": now()})
        log(f"    -> {res.get('status')} {res.get('error', '')} {res.get('elapsed_sec', '')}s "
            f"psnr0={((res.get('first_frame_check') or {}).get('psnr_db'))}")
        write_json(OUT / "status.json", summarize(rows))
    summary = summarize(rows)
    write_json(OUT / "status.json", summary)
    log(f"run end elapsed={time.time() - t0:.0f}s by_status={summary['by_status']}")
    return 0


# --------------------------------------------------------------------------- family backends
# SunRisesAndSets/lighting: where the case's local orbit frame sits in the Unity scene, and how
# the paper's conditioning frame (reference/ref.png) was rendered. Recovered by render-and-compare
# against ref.png (single-frame renders, coarse grid + Nelder-Mead over the free 6-DoF camera,
# then identified exactly): ref.png is the scene's own CameraOrbitAgent start view --
# lookAtTarget = TrainingArea at Unity (0.3154, 1.28, 0.54422), cameraDistance 3, cameraYaw 45,
# cameraPitch 20, the Main Camera's 60 deg vertical FOV, URP post-processing and the global
# Volume (SampleSceneProfile: tonemapping, bloom, vignette) on -- with the sun after
# DayAndNight's first 12 FixedUpdate steps (rotationSpeed 20 deg/s x 0.02 s: Euler x 63.4 ->
# 68.2). That pose at 1280x1280 matches ref.png at 30.9 dB (free-camera refinement: +0.06 dB;
# the scene's original sun 63.4 deg: 26.6 dB; post-processing off: ~21.4 dB; scene file of
# 82c7d4c, larger table and extra props: ~22 dB). The camera looks at the TrainingArea pivot,
# not the table axis (0.63 m away), so the orbit turns about that pivot: a look-at orbit about
# the table axis through the frame-0 position scores 20.4 dB at frame 0.
LIGHTING_NATIVE_UNITY = {"pivot_x": 0.3154, "pivot_z": 0.54422, "sun_euler_x": 68.2, "sun_euler_y": 330.0,
                         "sun_euler_z": 0.0, "post_processing": 1.0, "volume": 1.0}


def _orbit_prepare(doc: dict, row: dict, action: dict) -> None:
    import yaml
    from evaluation.camera_orbit import validate_orbit

    traj = yaml.safe_load((SUITE / row["path"] / "trajectory.yaml").read_text(encoding="utf-8"))
    validate_orbit(traj)
    g = traj["geometry"]
    if any(float(v) != 0.0 for v in g["origin_m"]):
        raise ValueError("orbit replay expects origin_m = [0, 0, 0] (the local frame sits on the orbit axis)")
    times = np.arange(doc["frames"]) / doc["fps"]
    phase = np.full(times.shape, float(g["initial_phase_deg"]))
    for seg in traj["segments"]:
        dt = np.clip(times - float(seg["start_sec"]), 0, float(seg["end_sec"]) - float(seg["start_sec"]))
        phase += dt * float(seg["values"]["orbit_phase_rate_deg_s"])
    doc["frame_phase_deg"] = [float(v) for v in phase]
    doc["vertical_fov_deg"] = float(g["vertical_fov_deg"])
    params = {"radius_m": float(g["radius_m"]), "eye_height_m": float(g["eye_height_m"]),
              "look_at_height_m": float(g["look_at_m"][1]), **LIGHTING_NATIVE_UNITY}
    doc["param_names"] = list(params)
    doc["param_values"] = [float(v) for v in params.values()]


def _orbit_export(raw: Path, art: Path, row: dict, spec: SceneSpec) -> dict[str, Any]:
    import yaml
    from evaluation.camera_orbit import camera_intrinsics, orbit_poses

    traj = yaml.safe_load((SUITE / row["path"] / "trajectory.yaml").read_text(encoding="utf-8"))
    cal = json.loads((raw / "calibration.json").read_text(encoding="utf-8"))
    frames = [json.loads(s) for s in (raw / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    vec = lambda v: np.array([v["x"], v["y"], v["z"]], dtype=np.float64)  # noqa: E731
    origin = vec(cal["originUnity"])
    reflect = np.diag([1.0, 1.0, -1.0])
    actual = np.tile(np.eye(4), (len(frames), 1, 1))
    for i, f in enumerate(frames):
        actual[i, :3, 3] = reflect @ (vec(f["positionUnity"]) - origin)
        actual[i, :3, :3] = reflect @ np.column_stack((vec(f["rightUnity"]), -vec(f["upUnity"]), vec(f["forwardUnity"])))
    times = np.array([f["canonicalTimeSec"] for f in frames])
    expected = orbit_poses(traj, times)
    err = float(np.max(np.abs(actual - expected)))
    if err > 2e-5:
        raise RuntimeError(f"Unity/C2W mismatch vs analytic orbit: {err}")
    np.save(art / "poses.npy", actual, allow_pickle=False)
    np.save(art / "intrinsics.npy", camera_intrinsics(traj, cal["width"], cal["height"]), allow_pickle=False)
    for name in ("calibration.json", "frames.jsonl", "scene_dependencies.json"):
        if (raw / name).is_file():
            shutil.copyfile(raw / name, art / name)
    return {"unity_c2w_max_abs_error": err, "pose_convention": "right_handed_y_up_opencv_c2w (Unity z reflected)"}


# ---- shared GT helpers
def _transforms(objects: dict[str, list[list[float]]], extra: dict | None = None) -> dict:
    out = {"schema_version": 1, "objects": {k: {"frames": list(range(len(v))), "positions_world": v}
                                           for k, v in objects.items()}}
    if extra:
        out.update(extra)
    return out


def _gt_rows(raw: Path, name: str = "gt.json") -> tuple[dict, list[dict]]:
    data = json.loads((raw / name).read_text(encoding="utf-8"))
    rows = data["frames"] if isinstance(data, dict) else data
    return (data if isinstance(data, dict) else {}), rows


def _save_camera(art: Path, rows: list[dict], intr_name: str = "intrinsics.npy") -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray([r["c2w"] for r in rows], dtype=np.float64).reshape(-1, 4, 4)
    intr = np.asarray([[r["fx"], r["fy"], r["cx"], r["cy"]] for r in rows], dtype=np.float64)
    np.save(art / "poses.npy", poses, allow_pickle=False)
    np.save(art / intr_name, intr, allow_pickle=False)
    return poses, intr


def _plane_calibration(raw: Path, art: Path, rows: list[dict], poses: np.ndarray, intr: np.ndarray, kind: str) -> None:
    from tools.run_unity_table_scenes import _export_planar_calibration

    _export_planar_calibration(art, raw / "frames/frame_0000.png", rows, poses, intr, kind)


def _write(art: Path, name: str, obj: Any) -> None:
    (art / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


# ---- camera-yaw pose synthesis (prompt-scene cases without a scene camera rig)
# RacingCar/prompt_third: the core poses.npy frame 0 IS the conditioning frame's camera pose,
# but its intrinsics.npy (fy=565 px, i.e. CarCamera.baseFOV 65 deg) is not: the core was
# recorded through Day.unity's Main Camera + CarCamera, whose field of view starts at the
# scene's 35 deg and eases towards 65 deg (Lerp by deltaTime*5 per LateUpdate), and frame 0
# is the first eased step (35 + 30 * 0.25 = 42.5 deg, the same step that moved the camera
# from the scene's (0, 2, 10) to P[0]). Core P[0] at 42.5 deg matches ref.png at 32.9 dB
# (FOV scan 41.5..44 deg, sharp optimum; a 6-DoF Nelder-Mead refinement around it found no
# better pose). The frame-0 camera (pose and FOV) is then yawed about the car.
RACING_THIRD_POSE0 = dict(fovy=42.5)


def _yaw_pose_prepare(third_person: bool, pose0: dict | None = None) -> Callable[[dict, dict, dict], None]:
    """Per-frame camera poses: the core reference's frame-0 pose, yawed about a
    vertical axis by the integrated yaw_rate (in place in first person, around
    the character in third person), as the core case's trajectory describes."""
    def prepare(doc: dict, row: dict, action: dict) -> None:
        core = case_dir(row["conditioning_case"]) / "reference/artifacts"
        P = np.load(core / "poses.npy").astype(np.float64)
        K = np.load(core / "intrinsics.npy").astype(np.float64)
        p0 = P[0]
        det = float(np.linalg.det(p0[:3, :3]))
        right, fwd, pos = p0[:3, 0], p0[:3, 2], p0[:3, 3]
        up = p0[:3, 1] if det > 0 else -p0[:3, 1]   # prompt cases store raw Unity axes
        if pose0 is not None and "pos" in pose0:   # pose0: optional pos/yaw/pitch and/or fovy overrides
            y_, p_ = math.radians(pose0["yaw"]), math.radians(pose0["pitch"])
            fwd = np.array([math.sin(y_) * math.cos(p_), -math.sin(p_), math.cos(y_) * math.cos(p_)])
            right = np.array([math.cos(y_), 0.0, -math.sin(y_)])
            up = np.cross(fwd, right)
            pos = np.asarray(pose0["pos"], dtype=np.float64)
        pivot = pos.copy()
        if third_person:
            x, z = P[:, 0, 3], P[:, 2, 3]
            A = np.c_[2 * x, 2 * z, np.ones_like(x)]
            sol = np.linalg.lstsq(A, x ** 2 + z ** 2, rcond=None)[0]
            pivot = np.array([sol[0], pos[1], sol[1]])
        fy = float(K[1, 1]) if K.ndim == 2 and K.shape == (3, 3) else float(np.asarray(K).reshape(-1, 4)[0, 1])
        fovy = math.degrees(2 * math.atan(doc["height"] / 2 / fy))
        if pose0 is not None and "fovy" in pose0:
            fovy = float(pose0["fovy"])
        frame_pos, frame_rot = [], []
        for yaw in doc["frame_yaw_deg"]:
            a = math.radians(yaw)
            R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            p = pivot + R @ (pos - pivot)
            M = np.column_stack((R @ right, R @ up, R @ fwd))
            frame_pos += [float(v) for v in p]
            frame_rot += [float(v) for v in _quat_xyzw(M)]
        doc["frame_pos"], doc["frame_rot"] = frame_pos, frame_rot
        doc["vertical_fov_deg"] = fovy
        doc["param_names"] += ["pivot_x", "pivot_z", "third_person"]
        doc["param_values"] += [float(pivot[0]), float(pivot[2]), 1.0 if third_person else 0.0]
    return prepare


def _quat_xyzw(M: np.ndarray) -> np.ndarray:
    m = M
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def _fixed_camera_prepare(pos: tuple[float, float, float], yaw: float, pitch: float, fovy: float,
                          chain: Callable | None = None) -> Callable[[dict, dict, dict], None]:
    """A fixed camera (Unity position + Euler(pitch, yaw, 0) + vertical FOV)."""
    def prepare(doc: dict, row: dict, action: dict) -> None:
        if chain:
            chain(doc, row, action)
        y, p = math.radians(yaw), math.radians(pitch)
        fwd = np.array([math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])
        right = np.array([math.cos(y), 0.0, -math.sin(y)])
        up = np.cross(fwd, right)
        doc["frame_pos"] = [float(v) for v in pos]
        doc["frame_rot"] = [float(v) for v in _quat_xyzw(np.column_stack((right, up, fwd)))]
        doc["vertical_fov_deg"] = float(fovy)
    return prepare


# Fixed cameras recovered from the core reference's frame 0 (the shipped scenes'
# follow cameras do not reproduce those views). Fitted by projecting the scene's
# known static geometry (tools/fit: chamfer least squares; median residual noted).
# MirrorRollingBall/lpips_collision: 47 yellow-ball centres, 0.30 px median residual.
# The core lpips_collision video was rendered through the scene's "Main Camera", whose URP
# camera data has post-processing on (SampleSceneProfile: Neutral tonemapping, bloom,
# vignette 0.2) -- hence its darker, desaturated, highlight-compressed look. Camera.main in
# the replay Player resolves to BootstrapSpectatorCamera's runtime-added camera, which has
# no post-processing (as in the prompt_first/third core views), so the spec captures from
# camera_name="Main Camera". Pose then re-refined by render-and-compare (Nelder-Mead on
# single-frame renders vs the core frame 0): 34.3 dB (was 18.5 dB without post-processing).
MIRROR_LPIPS_CAMERA = dict(pos=(-12.07204, 9.99295, -12.38425), yaw=46.47564, pitch=26.04596, fovy=70.62146)


# BallRoller/ctrl: arena wall corners, cube, dome and rack balls, 5.5 px RMS (7 points),
# refined by render-and-compare to 25.3 dB; the white ball's frame-0 position is the
# back-projection of its image centre onto the floor (ball radius 0.5).
BALLROLLER_CAMERA = dict(pos=(-7.08822, 3.66526, -1.12969), yaw=88.61807, pitch=13.87421, fovy=49.43086)
BALLROLLER_PLAYER_INIT = {"target_init_x": 3.312, "target_init_y": 0.488, "target_init_z": -2.134}


def _generic_prepare(target: str = "", force_source: str = "", disable: list[str] | None = None,
                     track: list[str] | None = None, camera: str = "", params: dict | None = None,
                     chain: Callable | None = None) -> Callable[[dict, dict, dict], None]:
    def prepare(doc: dict, row: dict, action: dict) -> None:
        if chain:
            chain(doc, row, action)
        doc["target_object"] = target
        doc["force_source"] = force_source
        doc["camera_name"] = camera
        doc["disable_types"] = list(disable or [])
        doc["track_objects"] = list(track or [])
        for k, v in (params or {}).items():
            doc["param_names"].append(k)
            doc["param_values"].append(float(v))
    return prepare


# ---- exports
def _garden_export(raw: Path, art: Path, row: dict, spec: SceneSpec) -> dict:
    data = json.loads((raw / "captured_data.json").read_text(encoding="utf-8"))
    rows = data["frames"]
    poses = np.asarray([np.asarray(r["c2w"], dtype=np.float64).reshape(4, 4) for r in rows])
    np.save(art / "poses.npy", poses, allow_pickle=False)
    np.save(art / "intrinsics.npy", np.asarray([[r["fx"], r["fy"], r["cx"], r["cy"]] for r in rows]), allow_pickle=False)
    objs = {k: [list(map(float, v))] * len(rows) for k, v in data["anchors"].items()}
    if "Main tree" in objs:
        objs["tree"] = objs["Main tree"]
    _write(art, "object_transforms.json", _transforms(objs))
    shutil.copyfile(raw / "captured_data.json", art / "unity_capture_log.json")
    return {}


BALL_LABELS = {"Ball_0": "white ball", "Ball_1": "yellow ball", "Ball_2": "red ball"}


def _billiard_export(kinematics: bool) -> Callable:
    def export(raw: Path, art: Path, row: dict, spec: SceneSpec) -> dict:
        _, rows = _gt_rows(raw)
        poses, intr = _save_camera(art, rows)
        objs = {label: [r["balls"][src] for r in rows] for src, label in BALL_LABELS.items()}
        _write(art, "object_transforms.json", _transforms(objs))
        if kinematics:
            _write(art, "object_masses.json", {label: 0.1701 for label in objs})
            _plane_calibration(raw, art, rows, poses, intr, "billiard")
            if (raw / "impulses.jsonl").is_file():
                shutil.copyfile(raw / "impulses.jsonl", art / "applied_impulses.jsonl")
        shutil.copyfile(raw / "gt.json", art / "unity_gt.json")
        return {}
    return export


def _racing_export(kinematics: bool) -> Callable:
    def export(raw: Path, art: Path, row: dict, spec: SceneSpec) -> dict:
        data, rows = _gt_rows(raw)
        poses, intr = _save_camera(art, rows)
        _write(art, "object_transforms.json", _transforms({"car": [r["car_pos"] for r in rows]}))
        if kinematics:
            _write(art, "object_masses.json", {"car": float(data["car_mass_kg"])})
            _plane_calibration(raw, art, rows, poses, intr, "racing")
        shutil.copyfile(raw / "gt.json", art / "unity_gt.json")
        return {}
    return export


def _generic_export(labels: dict[str, str] | None = None, intr_name: str = "intrinsics.npy",
                    masses: dict[str, float] | None = None, target_label: str = "") -> Callable:
    def export(raw: Path, art: Path, row: dict, spec: SceneSpec) -> dict:
        data, rows = _gt_rows(raw)
        _save_camera(art, rows, intr_name)
        objs: dict[str, list] = {}
        if target_label and rows and "target_pos" in rows[0]:
            objs[target_label] = [r["target_pos"] for r in rows]
        for src in (rows[0].get("objects", {}) if rows else {}):
            objs[(labels or {}).get(src, src)] = [r["objects"][src]["pos"] for r in rows]
        _write(art, "object_transforms.json", _transforms(objs))
        if masses is not None:
            m = dict(masses)
            if target_label and data.get("target_mass_kg"):
                m[target_label] = float(data["target_mass_kg"])
            _write(art, "object_masses.json", m)
        shutil.copyfile(raw / "gt.json", art / "unity_gt.json")
        return {}
    return export


def _chess_export(raw: Path, art: Path, row: dict, spec: SceneSpec) -> dict:
    _, rows = _gt_rows(raw)
    _save_camera(art, rows, "intrinsics_t4.npy")
    objs = {"sphere": [r["anchors"]["Sphere"] for r in rows if "Sphere" in r["anchors"]],
            "person": [r["player_pos"] for r in rows]}
    for k in ("King_Gold", "King_Wood", "King_Metal", "King_Marble"):
        if k in rows[0]["anchors"]:
            objs[k] = [r["anchors"][k] for r in rows]
    _write(art, "object_transforms.json", _transforms(objs))
    shutil.copyfile(raw / "gt.json", art / "unity_gt.json")
    return {}


def _liquid_post(d: Path, spec: SceneSpec, n_native: int) -> dict:
    """Keep the native 1280x720@30 render and derive the core-format reference
    (832x464 @ 16 fps) with the evaluator's own normalization."""
    from evaluation.lite.normalization import normalize_liquid_video

    native = d / "reference_native.mp4"
    (d / "reference.mp4").replace(native)
    normalize_liquid_video(native, d / "reference.mp4")
    return {"native_video": probe(native), "normalization": "evaluation.lite.normalization.normalize_liquid_video"}


BALLROLLER_OBJECTS = {"Obstacle_FixedRamp": "ramp", "Obstacle_FixedDome": "dome", "Obstacle_MovableCube": "cube"}
# The replay path (BottleController.SetInput per FixedUpdate) is implemented and runs, but
# the committed LiquidContainer scene is not the world of the core reference: the core
# fluid_rest video shows a coastal-Mediterranean HDRI sky and a checkerboard floor with
# coloured cross markers, and neither asset (nor a scene referencing them) exists in
# WorldModelBenchScenePrivate (all commits). The committed scene renders a different
# world (core-trajectory self-check: frame-0 PSNR 8.3 dB), so its renders would not
# start from the published conditioning frame. Needed: the scene revision/assets used
# for the core recording (HDRI + floor material + camera placement), then unblock.

DOMINOES = [f"Domino_S{s:02d}_D{k:02d}" for s in range(12) for k in range(5)]


def register_specs() -> None:
    def add(family: str, cases: list[str], **kw: Any) -> None:
        for c in cases:
            SPECS[f"{family}/{c}"] = SceneSpec(key=f"{family}/{c}", **kw)

    garden = dict(project="GameScenes/WanderingGarden", scene="Assets/Scenes/Garden/GardenScene.unity",
                  product="WanderingGarden", fps=15, width=1280, height=720, mode="yaw", export=_garden_export)
    billiard = dict(project="GameScenes/BilliardTemplate", scene="Assets/Vagho/Billiard/Scenes/CaromBilliard.unity",
                    product="BilliardGame", width=1280, height=720)
    racing = dict(project="GameScenes/RacingCar", scene="Assets/ARCADE - FREE Racing Car/Scenes/Day.unity",
                  product="RacingCar", width=1280, height=720, fps=30)
    mirror = dict(project="GameScenes/MirrorRollingBall", scene="Assets/Scenes/SampleScene.unity",
                  product="MirrorRollingBall", width=1280, height=720)
    chess = dict(project="GameScenes/ChessBoardwithSphere", scene="Assets/Scenes/Chess Demo.unity",
                 product="ChessBoard", fps=30)
    ball_disable = ["PlayerController", "BallAgent", "DecisionRequester", "CameraSensorComponent"]

    # camera_yaw ---------------------------------------------------------------
    add("camera_yaw", ["WanderingGarden/scene_360_in"], **garden,
        runtime_args=["-wmeval-cam-pos", "3.277000,1.600000,69.049000", "-wmeval-cam-yaw", "298.2"])
    add("camera_yaw", ["WanderingGarden/scene_360_out"], **garden,
        # The core 360_out conditioning frame is frame 75 of the bridge sweep: start
        # 75 frames later in simulated time so animated water/foliage match too.
        runtime_args=["-wmeval-cam-pos", "3.277000,1.600000,69.049000", "-wmeval-cam-yaw", "118.2",
                      "-wmeval-warmup-frames", "95"])
    add("camera_yaw", ["BilliardGame/scene_360_in"], **billiard, fps=15, mode="yaw",
        runtime_args=["-wmeval-view", "third", "-wmeval-camera-mode", "schedule", "-wmeval-start-yaw-deg", "0",
                      "-wmeval-no-shot", "-wmeval-no-metric-anchors", "-wmeval-cue-ball-index", "1"],
        export=_billiard_export(False))
    for view, third in (("prompt_first", False), ("prompt_third", True)):
        add("camera_yaw", [f"MirrorRollingBall/{view}"], **mirror, fps=30, mode="camera_poses",
            prepare=_generic_prepare(disable=ball_disable + ["CameraController"], track=["Player"],
                                     chain=_yaw_pose_prepare(third)),
            export=_generic_export(labels={"Player": "ball"}))
        # The prompt references show the Feb 2026 chess scene (before cf2af02 added the hourglass,
        # candles, statues and player on 2026-04-21), so these render from that revision.
        # Environment lighting: the scene's chrome-sphere ReflectionProbe is Realtime, but the
        # batch-mode Player (Camera.Render only) never refreshes it, so it kept the baked
        # ReflectionProbe-2.exr from LightingData.asset -- baked by the author with the
        # SkySeries cloud skybox textures, which are not in the repo (the skybox renders flat
        # grey here and in the core references). The core frames show the live probe: grey
        # sky, captured objects without environment reflections (black), no character.
        # -wmeval-reflection-probes realtime re-renders it that way (WMEvalReplayCapture).
        # warmup_frames=60: the core capture's 2 s warmup (meta.json warmup_s), so the third-
        # person character has settled onto the board in its idle pose at frame 0. First person:
        # the camera sits inside the (static, idling) character's head, whose mesh would show
        # at some yaws; the core first-person video never shows the body, so hide it.
        add("camera_yaw", [f"ChessBoard/{view}"], **{**chess, "product": "ChessBoardFeb"}, source_rev="93ec507",
            asset_note="ChessBoardwithSphere at 93ec507 (2026-02-10) with all its LFS assets; "
                       "sphere reflection probe re-rendered live (no cloud skybox textures)",
            width=1280, height=720, mode="camera_poses",
            runtime_args=["-wmeval-reflection-probes", "realtime", "-wmeval-probe-env", "black",
                          "-wmeval-probe-hide", "PlayerArmature"]
                         + ([] if third else ["-wmeval-hide-renderers", "PlayerArmature"]),
            prepare=_generic_prepare(disable=["FreeCameraController", "CinemachineBrain", "PlayerAgent", "DecisionRequester"],
                                     track=["PlayerArmature", "Sphere"], params={"warmup_frames": 60},
                                     chain=_yaw_pose_prepare(third)),
            export=_generic_export(labels={"PlayerArmature": "person", "Sphere": "sphere"}))
        # The prompt references predate the WMEvalOrbitAnchor probe cube in Day.unity,
        # and the first-person one shows no car body (camera sits inside the car).
        add("camera_yaw", [f"RacingCar/{view}"], **racing, mode="camera_poses",
            runtime_args=["-wmeval-capture", "-wmeval-view", "third" if third else "first",
                          "-wmeval-hide", "WMEvalOrbitAnchor"]
                         + ([] if third else ["-wmeval-hide-renderers", "Free Racing Car"]),
            prepare=_yaw_pose_prepare(third, RACING_THIRD_POSE0 if third else None), export=_racing_export(False))

    # camera_orbit -------------------------------------------------------------
    add("camera_orbit", ["SunRisesAndSets/lighting"], project="GameScenes/SunRisesAndSets",
        scene="Assets/WMEval/SampleScene_84a06a4.unity", product="SunRisesAndSets", fps=24, width=1280, height=1280,
        mode="orbit_phase", kind="editor", editor_method="WMEvalOrbitReplay.Run",
        runtime_args=["-wmeval-scene", "Assets/WMEval/SampleScene_84a06a4.unity"],
        frames_override=lambda d: int(round(d * 24)) + 1,   # endpoint pose rendered too (241), closure check
        encode_frames=lambda n: n - 1,                      # video carries 240 frames (10 s at 24 fps)
        asset_note="SunRisesAndSets scene file pinned to 84a06a4 (2026-02-17), the revision of the paper's "
                   "conditioning frame; native CameraOrbitAgent rig, sun Euler x 68.2, post-processing on",
        prepare=_orbit_prepare, export=_orbit_export)

    # ego_vehicle / cue_ball ---------------------------------------------------
    add("ego_vehicle", ["RacingCar/kinematics_third"], **racing, mode="vehicle_controls",
        runtime_args=["-wmeval-capture", "-wmeval-view", "third"], export=_racing_export(True))
    add("cue_ball", ["BilliardGame/kinematics_first"], **billiard, fps=30, mode="ball_impulses",
        runtime_args=["-wmeval-view", "first", "-wmeval-camera-mode", "static", "-wmeval-control-mode", "schedule",
                      "-wmeval-cue-ball-index", "0"],
        export=_billiard_export(True))

    # ball_force ---------------------------------------------------------------
    add("ball_force", ["BallRoller/ctrl"], project="GameScenes/BallRoller",
        scene="Assets/UnityTechnologies/RollABall/Roll-a-ball.unity", product="BallRoller", fps=30, width=1280,
        height=720, mode="rigidbody_force", first_frame_ref="reference/reference.mp4",
        prepare=_generic_prepare(target="Player", force_source="BallAgent.speed",
                                 disable=ball_disable + ["CameraController"],
                                 track=list(BALLROLLER_OBJECTS) + [f"Physics Ball {i}" for i in range(1, 16)]
                                 + ["North Wall", "South Wall", "East Wall", "West Wall"],
                                 params=BALLROLLER_PLAYER_INIT,
                                 chain=_fixed_camera_prepare(**BALLROLLER_CAMERA)),
        export=_generic_export(labels=BALLROLLER_OBJECTS, target_label="white ball",
                               masses={"ramp": 1.0, "dome": 1.0, "cube": 1.0}))
    add("ball_force", ["MirrorRollingBall/lpips_collision"], **mirror, fps=60, mode="rigidbody_force",
        first_frame_ref="reference/reference.mp4",
        prepare=_generic_prepare(target="Player", force_source="BallAgent.forceMultiplier",
                                 disable=ball_disable + ["CameraController"], camera="Main Camera",
                                 chain=_fixed_camera_prepare(**MIRROR_LPIPS_CAMERA)),
        export=_generic_export(target_label="mirror ball", masses={}))

    # beam_domino --------------------------------------------------------------
    add("beam_domino", ["BeamDomino/causal"], project="GameScenes/BeamDomino",
        scene="Assets/Scenes/BeamDomino/BeamDomino.unity", product="BeamDomino", fps=60, width=1280, height=720,
        mode="agent_heuristic", first_frame_ref="reference/reference.mp4",
        # Camera: BeamDominoAgent's own follow camera, untouched (core-trajectory self-check:
        # frame 0 at 28 dB, 34-39 dB until the push; posing frame 0 at PositionCamera()'s
        # logged start pose scored worse, 19.6 dB, so the scene camera is left alone).
        prepare=_generic_prepare(target="FloorBall", track=["RoofBall", "Roof", "Pillar"] + DOMINOES),
        export=_generic_export(target_label="FloorBall", masses={}))

    # liquid_container ---------------------------------------------------------
    add("liquid_container", ["LiquidContainer/fluid_rest", "LiquidContainer/fluid_step_hold",   # the last three
        "LiquidContainer/fluid_impulse_decay", "LiquidContainer/fluid_chirp_strafe"],      # for selfcheck only
        project="GameScenes/LiquidContainer",
        scene="Assets/Scenes/LiquidContainer.unity", product="LiquidContainer", fps=30, width=1280, height=720,
        mode="bottle_input", first_frame_ref="reference/reference.mp4", post_encode=_liquid_post,
        # freeze_rotation / use_gravity (2026-10-21 dataset audit): with the committed scene's
        # Rigidbody (no rotation constraints, gravity off) a bottle sliding along the camera
        # axis is tipped ~52 deg by floor friction and stays tipped, and an upward input leaves
        # it hovering; the core recordings stay upright and fall back after the jump. The
        # core-trajectory self-checks (selfcheck --family liquid_container) compare both.
        prepare=_generic_prepare(target="Bottle", params={"freeze_rotation": 1, "use_gravity": 1}),
        export=_generic_export(target_label="bottle"))

    # character ----------------------------------------------------------------
    # walk_third was recorded from the April scene on a machine that only had the Feb-era LFS
    # assets (the later props were missing); reproducing that state matches it at 99 dB.
    add("character", ["ChessBoard/walk_third"], **{**chess, "product": "ChessBoardFebAssets"}, lfs_rev="93ec507",
        asset_note="current ChessBoardwithSphere with only the LFS assets present at 93ec507 (2026-02-10)",
        width=640, height=360, mode="character",
        export=_chess_export)


register_specs()

if __name__ == "__main__":
    raise SystemExit(main())
