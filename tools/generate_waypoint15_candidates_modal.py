#!/usr/bin/env python3
"""Generate Waypoint-1.5 candidate rollouts for WMEval Lite cases on Modal.

Unity-free: every generation condition comes from the case bundle itself.

* seed frame  = ``reference/ref.png`` when the case ships one, else frame 0 of
  the bundled reference video (resized straight to Waypoint's 1280x720 input,
  the same way the paper's Waypoint runs were seeded)
* actions     = ``reference/artifacts/actions.txt`` translated to Waypoint
  ``CtrlInput`` steps (Waypoint-1.5 emits 4 frames per control step at 60 fps,
  so one step is 66.7 ms). Keyboard ops map to button IDs, ``ArrowLeft`` /
  ``ArrowRight`` to a constant mouse yaw (0.15 per step, the value the paper's
  Waypoint runs used for "rotate left"). Waypoint has no text channel and no
  view switch, so ``setview`` ops are ignored and the prompt is unused.
* duration    = the reference video's duration; shorter scripts are padded with
  idle steps so evaluators see the same clip length as the reference.

Two trees are written to the ``WMEval-lite-candidates`` volume:

* ``waypoint15/<Scene>/<variant>/<seed>.mp4``  -- the evaluated candidate, in
  the runner's ``<model_root>/<scene>/<variant>/<seed>.mp4`` layout, normalised
  to the reference video's fps, resolution and duration (kinematics and object
  consistency compare frame-by-frame in the reference pixel frame; the
  ``candidate_normalization: none`` cases have no other normalisation step)
* ``waypoint15_native/<Scene>/<variant>/<seed>.mp4`` + ``<seed>.json``  -- the
  raw 60 fps 1280x720 output and a provenance manifest (controls, seed, model)

Cases whose ``actions.txt`` is missing or purely descriptive get an explicit
control program in ``CASE_OVERRIDES`` (derived from the bundle's own notes).

Usage::

    # smoke test: one case, one seed
    modal run tools/generate_waypoint15_candidates_modal.py --only ChessBoard/walk_third --seeds 42

    # every generatable case x 5 seeds, one H100 container per case
    modal run --detach tools/generate_waypoint15_candidates_modal.py

    # copy the paper's existing Waypoint rollouts (prompt_first x2, lighting)
    # from WMEval-lighting-repro into the same layout
    modal run tools/generate_waypoint15_candidates_modal.py::import_existing
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = REPO_ROOT / "eval_sets" / "wmeval_lite_v1" / "cases"
REMOTE_CASES = "/workspace/cases"

MODEL_NAME = "waypoint15"
MODEL_ID = "Overworld/Waypoint-1.5-1B"
CANDIDATES_VOLUME = "WMEval-lite-candidates"
LIGHTING_REPRO_VOLUME = "WMEval-lighting-repro"
HF_CACHE_VOLUME = "wm-hf-cache"
CONTAINER_CAND = "/cand"
CONTAINER_REPRO = "/repro"
CONTAINER_HF_CACHE = "/cache/hf"

NATIVE_W, NATIVE_H, NATIVE_FPS = 1280, 720, 60
FRAMES_PER_STEP = 4
STEP_MS = 1000.0 * FRAMES_PER_STEP / NATIVE_FPS  # 66.67 ms
YAW_PER_STEP = 0.15  # mouse dx per step; the paper's Waypoint "turn left" value
DEFAULT_SEEDS = ["42", "1", "2", "3", "4"]

# Waypoint button IDs are Owl-Control keycodes. The public example
# (world_engine/examples/gen_sample.py) uses 65/68/83 for A/D/S and 32 for
# Space, i.e. Windows virtual-key codes; W and Left-Shift follow that table.
KEY_IDS = {
    "KeyW": 87,
    "KeyA": 65,
    "KeyS": 83,
    "KeyD": 68,
    "Space": 32,
    "ShiftLeft": 160,
}
ARROW_MOUSE = {
    "ArrowLeft": (-YAW_PER_STEP, 0.0),
    "ArrowRight": (YAW_PER_STEP, 0.0),
    "ArrowUp": (0.0, -YAW_PER_STEP),
    "ArrowDown": (0.0, YAW_PER_STEP),
}

# Control programs for cases whose actions.txt is absent or descriptive only.
# Same op grammar as actions.txt, plus a multi-key "keys" form for chords.
CASE_OVERRIDES: dict[str, list[dict]] = {
    # actions.txt is a comment-only frame table (30 fps, 90 frames):
    #   0-24 walk fwd | 25-44 diagonal (move.x .8, .6; jump at frame 30)
    #   45-69 sprint fwd | 70-89 turn back left (move.x -.5) + walk fwd
    "ChessBoard/walk_third": [
        {"op": "hold", "keys": ["KeyW"], "ms": 833},
        {"op": "hold", "keys": ["KeyW", "KeyD"], "ms": 167},
        {"op": "hold", "keys": ["KeyW", "KeyD", "Space"], "ms": 67},
        {"op": "hold", "keys": ["KeyW", "KeyD"], "ms": 467},
        {"op": "hold", "keys": ["KeyW", "ShiftLeft"], "ms": 833},
        {"op": "hold", "keys": ["KeyW", "KeyA"], "ms": 667},
    ],
    # case.yaml: "deterministic in-place 360-degree yaw sweep from a fixed
    # standing position" over the 10 s clip; no actions.txt in the bundle.
    "WanderingGarden/scene_first": [
        {"op": "hold", "key": "ArrowLeft", "ms": 10000},
    ],
    # Fixed top-down billiards; the cue strike is a scene event, not a player
    # action Waypoint can express. Idle input for the reference duration.
    "BilliardGame/prompt_first": [],
}

# Not generated here: reference-free lighting, imported from the paper run.
IMPORT_ONLY = {"SunRisesAndSets/lighting"}
# Cases to leave out (none: BeamDomino/causal is wired into lpips and
# action_controllability since #31).
SKIP_CASES: set[str] = set()

app = modal.App("wmeval-waypoint15-candidates")

candidates_vol = modal.Volume.from_name(CANDIDATES_VOLUME, create_if_missing=True)
repro_vol = modal.Volume.from_name(LIGHTING_REPRO_VOLUME, create_if_missing=False)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

CU128 = "https://download.pytorch.org/whl/cu128"

# Mirrors the paper's Waypoint harness image, with world_engine's current
# pyproject pins (torch 2.11 cu128, transformers>=5.3, diffusers, hf-hub>=1.7).
waypoint_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install(
        "huggingface_hub[cli,hf_transfer]>=1.7.0",
        "hf_transfer",
        "packaging",
        "ninja",
        "wheel",
        "setuptools",
    )
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        extra_options=f"--index-url {CU128}",
    )
    .pip_install("triton==3.6.0")
    .pip_install(
        "imageio[pyav]",
        "opencv-python-headless==4.10.0.84",
        "numpy<2",
        "transformers>=5.3.0",
        "accelerate==1.12.0",
        "tensordict==0.10.0",
        "diffusers",
        "einops",
        "ftfy",
        "omegaconf",
        "PyYAML>=6,<7",
        "https://github.com/madebyollin/taehv/archive/7dc60ec6601af2e668e31bc70acc4cb3665e4c22.zip",
        "https://github.com/dropbox/gemlite/archive/ebfcc197d694acd4ca8dfaadf5fceb3570401df3.zip",
    )
    .pip_install(
        "git+https://github.com/Overworldai/world_engine.git",
        extra_options="--no-deps",
    )
    .env(
        {
            "HF_HUB_CACHE": CONTAINER_HF_CACHE,
            "HF_HOME": CONTAINER_HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCHINDUCTOR_CACHE_DIR": f"{CONTAINER_HF_CACHE}/inductor",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(str(CASES_DIR), REMOTE_CASES, copy=True, ignore=["**/rubrics/**", "**/*.npz"])
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("opencv-python-headless==4.10.0.84", "numpy<2", "PyYAML>=6,<7")
    .add_local_dir(str(CASES_DIR), REMOTE_CASES, copy=True, ignore=["**/rubrics/**", "**/*.npz"])
)


# ---------------------------------------------------------------------------
# Case inputs (pure functions over the bundle; no torch)
# ---------------------------------------------------------------------------
def _parse_actions_txt(text: str) -> list[dict]:
    """actions.txt is either one JSON array or JSONL ops; '#' lines are notes."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    body = "\n".join(lines)
    try:
        parsed = json.loads(body)
        if isinstance(parsed, list):
            return [op for op in parsed if isinstance(op, dict)]
    except json.JSONDecodeError:
        pass
    ops = []
    for ln in lines:
        ops.append(json.loads(ln))
    return ops


def _op_keys(op: dict) -> list[str]:
    if "keys" in op:
        return list(op["keys"])
    if "key" in op:
        return [op["key"]]
    return []


def compile_controls(ops: list[dict], total_steps: int) -> tuple[list[tuple[set[int], tuple[float, float]]], list[dict]]:
    """Translate actions.txt ops into per-step (buttons, mouse) tuples.

    Returns the step list (padded/truncated to ``total_steps``) and a compact
    segment log for the provenance manifest.
    """
    pressed: set[int] = set()
    steps: list[tuple[set[int], tuple[float, float]]] = []
    segments: list[dict] = []

    def emit(n: int, buttons: set[int], mouse: tuple[float, float], label: str) -> None:
        for _ in range(n):
            steps.append((set(buttons), mouse))
        segments.append({"steps": n, "buttons": sorted(buttons), "mouse": list(mouse), "label": label})

    for op in ops:
        kind = str(op.get("op", "")).lower()
        ms = float(op.get("ms", 0) or 0)
        n = max(1, int(round(ms / STEP_MS))) if ms > 0 else 0
        keys = _op_keys(op)
        if kind == "setview":
            continue  # Waypoint has no view mode; the seed frame decides
        if kind == "wait":
            emit(n, pressed, (0.0, 0.0), "wait")
        elif kind == "hold":
            mouse = (0.0, 0.0)
            buttons = set(pressed)
            for key in keys:
                if key in ARROW_MOUSE:
                    dx, dy = ARROW_MOUSE[key]
                    mouse = (mouse[0] + dx, mouse[1] + dy)
                elif key in KEY_IDS:
                    buttons.add(KEY_IDS[key])
                else:
                    raise ValueError(f"unmapped key {key!r} in op {op}")
            emit(n, buttons, mouse, "hold " + "+".join(keys))
        elif kind == "down":
            for key in keys:
                pressed.add(KEY_IDS[key])
            if n:
                emit(n, pressed, (0.0, 0.0), "down " + "+".join(keys))
        elif kind == "up":
            for key in keys:
                pressed.discard(KEY_IDS[key])
            if n:
                emit(n, pressed, (0.0, 0.0), "up " + "+".join(keys))
        else:
            raise ValueError(f"unknown op {kind!r} in {op}")

    scripted = len(steps)
    if scripted < total_steps:
        emit(total_steps - scripted, pressed, (0.0, 0.0), "pad_idle")
    elif scripted > total_steps:
        steps = steps[:total_steps]
        segments.append({"steps": scripted - total_steps, "label": "truncated_to_reference_duration"})
    return steps, segments


def _reference_video(case_dir: Path) -> Path | None:
    for name in ("unity_groundtruth.mp4", "reference.mp4"):
        path = case_dir / "reference" / name
        if path.is_file():
            return path
    return None


def probe_case(case_dir: Path) -> dict:
    """Seed source, reference fps/resolution/duration, and the control ops."""
    import cv2

    case_id = f"{case_dir.parent.name}/{case_dir.name}"
    ref_png = case_dir / "reference" / "ref.png"
    ref_video = _reference_video(case_dir)
    if ref_video is None:
        raise ValueError(f"{case_id}: no reference video to take duration/fps from")
    cap = cv2.VideoCapture(str(ref_video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or n_frames <= 0:
        raise ValueError(f"{case_id}: cannot probe {ref_video}")
    duration = n_frames / fps

    if case_id in CASE_OVERRIDES:
        ops = CASE_OVERRIDES[case_id]
        ops_source = "CASE_OVERRIDES"
    else:
        actions = case_dir / "reference" / "artifacts" / "actions.txt"
        ops = _parse_actions_txt(actions.read_text(encoding="utf-8")) if actions.is_file() else []
        ops_source = "reference/artifacts/actions.txt" if actions.is_file() else "none"

    total_steps = int(math.ceil(duration * NATIVE_FPS / FRAMES_PER_STEP))
    return {
        "case_id": case_id,
        "seed_source": "reference/ref.png" if ref_png.is_file() else f"frame0:{ref_video.name}",
        "seed_path": str(ref_png if ref_png.is_file() else ref_video),
        "reference_video": str(ref_video),
        "reference_fps": fps,
        "reference_frames": n_frames,
        "reference_width": width,
        "reference_height": height,
        "reference_duration_sec": duration,
        "total_steps": total_steps,
        "ops": ops,
        "ops_source": ops_source,
    }


def load_seed_frame_rgb(seed_path: str):
    import cv2
    import numpy as np

    if seed_path.lower().endswith(".png"):
        bgr = cv2.imread(seed_path, cv2.IMREAD_COLOR)
    else:
        cap = cv2.VideoCapture(seed_path)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read frame 0 of {seed_path}")
    rgb = cv2.cvtColor(cv2.resize(bgr, (NATIVE_W, NATIVE_H)), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def normalize_to_reference(native_mp4: str, out_mp4: str, fps: float, width: int, height: int, duration: float) -> None:
    """Resample the native 60 fps 1280x720 rollout to the reference's fps,
    resolution and duration so frame-indexed evaluators line up."""
    import subprocess

    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", native_mp4,
        "-vf", f"fps={fps:.6f},scale={width}:{height}:flags=lanczos",
        "-t", f"{duration:.4f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", out_mp4,
    ]
    subprocess.run(cmd, check=True)


class FfmpegWriter:
    """Pipe uint8 RGB frames (N, H, W, 3) into ffmpeg/libx264.

    Replaces imageio's pyav plugin, which re-assigns ``stream.width`` on
    every frame of a batch and therefore fails on PyAV >= 14 ("Cannot change
    width after codec is open").
    """

    def __init__(self, path: str, width: int, height: int, fps: int) -> None:
        import subprocess

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height
        self.frames_written = 0
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frames) -> None:
        import numpy as np

        arr = np.ascontiguousarray(frames, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[None]
        if arr.shape[1:] != (self.height, self.width, 3):
            raise RuntimeError(f"frame batch shape {arr.shape} != (N, {self.height}, {self.width}, 3)")
        assert self.proc.stdin is not None
        self.proc.stdin.write(arr.tobytes())
        self.frames_written += int(arr.shape[0])

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with {rc}")


def _write_model_yaml(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.yaml").write_text(
        "# Per-model candidate params picked up by tools/run_wmeval_lite.py run --model-root\n"
        "shadow_hud_preset: none  # Waypoint-1.5 renders no HUD overlay\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------
@app.function(
    image=waypoint_image,
    gpu="H100",
    timeout=3 * 3600,
    volumes={CONTAINER_CAND: candidates_vol, CONTAINER_HF_CACHE: hf_cache_vol},
    secrets=[hf_secret],
)
def generate_case(case_id: str, seeds: list[str], overwrite: bool = False) -> dict:
    """Generate every seed of one case in a single container (one model load,
    one torch.compile) and write native + normalised videos to the volume."""
    import os
    import time

    import numpy as np
    import torch
    from world_engine import CtrlInput, WorldEngine

    case_dir = Path(REMOTE_CASES) / case_id
    info = probe_case(case_dir)
    steps, segments = compile_controls(info["ops"], info["total_steps"])
    seed_rgb = load_seed_frame_rgb(info["seed_path"])
    seed_x4 = torch.from_numpy(np.repeat(seed_rgb[None], FRAMES_PER_STEP, axis=0))

    eval_root = Path(CONTAINER_CAND) / MODEL_NAME
    native_root = Path(CONTAINER_CAND) / f"{MODEL_NAME}_native"
    _write_model_yaml(eval_root)

    todo = []
    for seed in seeds:
        out_mp4 = eval_root / case_id / f"{seed}.mp4"
        if out_mp4.is_file() and not overwrite:
            print(f"[{case_id}/{seed}] exists, skipping", flush=True)
            continue
        todo.append(seed)
    if not todo:
        return {"case_id": case_id, "generated": [], "skipped": seeds}

    t0 = time.time()
    print(f"[{case_id}] loading {MODEL_ID} ...", flush=True)
    engine = WorldEngine(MODEL_ID, quant=None, device="cuda")
    load_sec = time.time() - t0

    results = []
    for seed in todo:
        tag = f"{case_id}/{seed}"
        native_mp4 = native_root / case_id / f"{seed}.mp4"
        native_mp4.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        engine.reset()
        t1 = time.time()
        writer = FfmpegWriter(str(native_mp4), NATIVE_W, NATIVE_H, NATIVE_FPS)
        try:
            engine.append_frame(seed_x4)
            writer.write(seed_x4.cpu().numpy())
            for i, (buttons, mouse) in enumerate(steps):
                ctrl = CtrlInput(button=set(buttons), mouse=[mouse[0], mouse[1]])
                frames = engine.gen_frame(ctrl=ctrl).cpu().numpy()
                writer.write(frames)
                if i == 0:
                    # first gen_frame == torch.compile; persist the inductor
                    # cache now so a later failure does not throw it away
                    hf_cache_vol.commit()
                    print(f"[{tag}] first step done in {time.time() - t1:.0f}s (incl. compile); frame batch {frames.shape}", flush=True)
                elif i % 30 == 0:
                    print(f"[{tag}] step {i}/{len(steps)}", flush=True)
        finally:
            writer.close()
        gen_sec = time.time() - t1

        out_mp4 = eval_root / case_id / f"{seed}.mp4"
        normalize_to_reference(
            str(native_mp4), str(out_mp4),
            fps=info["reference_fps"], width=info["reference_width"],
            height=info["reference_height"], duration=info["reference_duration_sec"],
        )
        manifest = {
            "model": MODEL_NAME,
            "model_id": MODEL_ID,
            "case_id": case_id,
            "seed": seed,
            "seed_source": info["seed_source"],
            "seed_resize": f"direct to {NATIVE_W}x{NATIVE_H}",
            "ops_source": info["ops_source"],
            "control_steps": len(steps),
            "frames_per_step": FRAMES_PER_STEP,
            "native": {"fps": NATIVE_FPS, "width": NATIVE_W, "height": NATIVE_H, "frames": FRAMES_PER_STEP * (len(steps) + 1)},
            "normalized_to_reference": {
                "fps": info["reference_fps"], "width": info["reference_width"],
                "height": info["reference_height"], "duration_sec": info["reference_duration_sec"],
            },
            "segments": segments,
            "yaw_per_step": YAW_PER_STEP,
            "key_ids": KEY_IDS,
            "timing_sec": {"model_load": round(load_sec, 1), "generation": round(gen_sec, 1)},
        }
        (native_root / case_id / f"{seed}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        candidates_vol.commit()
        hf_cache_vol.commit()  # persist the inductor cache
        print(f"[{tag}] done in {gen_sec:.0f}s -> {out_mp4}", flush=True)
        results.append({"seed": seed, "generation_sec": round(gen_sec, 1), "native_size": os.path.getsize(native_mp4), "out": str(out_mp4)})

    return {"case_id": case_id, "steps": len(steps), "model_load_sec": round(load_sec, 1), "generated": results}


@app.function(
    image=cpu_image,
    timeout=3600,
    volumes={CONTAINER_CAND: candidates_vol, CONTAINER_REPRO: repro_vol},
)
def import_existing_rollouts(overwrite: bool = False) -> list[str]:
    """Bring the paper's existing Waypoint rollouts into the same layout:
    prompt_first (RacingCar, MirrorRollingBall; normalised to the reference)
    and SunRisesAndSets/lighting (reference-free; kept native)."""
    import shutil

    eval_root = Path(CONTAINER_CAND) / MODEL_NAME
    native_root = Path(CONTAINER_CAND) / f"{MODEL_NAME}_native"
    _write_model_yaml(eval_root)
    done = []
    for scene in ("RacingCar", "MirrorRollingBall"):
        case_id = f"{scene}/prompt_first"
        info = probe_case(Path(REMOTE_CASES) / case_id)
        for seed in DEFAULT_SEEDS:
            src = Path(CONTAINER_REPRO) / "rollouts" / "prompt_first" / scene / f"{seed}.mp4"
            out = eval_root / case_id / f"{seed}.mp4"
            if out.is_file() and not overwrite:
                continue
            native = native_root / case_id / f"{seed}.mp4"
            native.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, native)
            normalize_to_reference(
                str(native), str(out), fps=info["reference_fps"], width=info["reference_width"],
                height=info["reference_height"], duration=info["reference_duration_sec"],
            )
            (native_root / case_id / f"{seed}.json").write_text(json.dumps({
                "model": MODEL_NAME, "case_id": case_id, "seed": seed,
                "origin": f"{LIGHTING_REPRO_VOLUME}:{src.relative_to(CONTAINER_REPRO)}",
                "note": "paper run (harness/waypoint_modal.py::lite, 150 steps left yaw)",
                "normalized_to_reference": {k: info[k] for k in ("reference_fps", "reference_width", "reference_height", "reference_duration_sec")},
            }, indent=2), encoding="utf-8")
            done.append(str(out))
    for seed in DEFAULT_SEEDS:
        src = Path(CONTAINER_REPRO) / "rollouts" / "lighting" / f"waypoint15_seed{seed}.mp4"
        out = eval_root / "SunRisesAndSets" / "lighting" / f"{seed}.mp4"
        if out.is_file() and not overwrite:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        (native_root / "SunRisesAndSets" / "lighting").mkdir(parents=True, exist_ok=True)
        (native_root / "SunRisesAndSets" / "lighting" / f"{seed}.json").write_text(json.dumps({
            "model": MODEL_NAME, "case_id": "SunRisesAndSets/lighting", "seed": seed,
            "origin": f"{LIGHTING_REPRO_VOLUME}:{src.relative_to(CONTAINER_REPRO)}",
            "note": "paper run (90 steps left yaw); reference-free dimension, kept native 60 fps 1280x720",
        }, indent=2), encoding="utf-8")
        done.append(str(out))
    candidates_vol.commit()
    return done


def generatable_cases() -> list[str]:
    cases = []
    for case_yaml in sorted(CASES_DIR.glob("*/*/case.yaml")):
        case_id = f"{case_yaml.parent.parent.name}/{case_yaml.parent.name}"
        if case_id in IMPORT_ONLY or case_id in SKIP_CASES:
            continue
        cases.append(case_id)
    return cases


@app.local_entrypoint()
def main(only: str = "", seeds: str = ",".join(DEFAULT_SEEDS), overwrite: bool = False, plan: bool = False):
    """Generate candidates. --only filters case ids by substring; --plan only
    prints the compiled control program per case (no GPU)."""
    seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
    cases = [c for c in generatable_cases() if not only or only in c]
    if not cases:
        print("no cases selected")
        return
    for case_id in cases:
        info = probe_case(CASES_DIR / case_id)
        steps, segments = compile_controls(info["ops"], info["total_steps"])
        print(
            f"{case_id}: seed={info['seed_source']} ref={info['reference_width']}x{info['reference_height']}"
            f"@{info['reference_fps']:.0f}fps {info['reference_duration_sec']:.2f}s -> {len(steps)} steps "
            f"(ops from {info['ops_source']}): "
            + "; ".join(f"{s['label']}x{s['steps']}" for s in segments)
        )
    if plan:
        return
    print(f"\nlaunching {len(cases)} case container(s) x {len(seed_list)} seed(s) on H100 ...")
    for result in generate_case.map(cases, kwargs={"seeds": seed_list, "overwrite": overwrite}):
        print(json.dumps(result))


@app.local_entrypoint()
def import_existing(overwrite: bool = False):
    for path in import_existing_rollouts.remote(overwrite=overwrite):
        print(path)
