#!/usr/bin/env python3
"""Prepare LingBot (Cam) and Matrix-Game 3.0 inputs for the prompt-scene cases.

Every ``prompt_scene_consistency`` case turns the camera left for 10 s at the
Unity reference's 40 deg/s: the canonical semantic trajectory
``experiments/protocol/trajectories/prompt_camera_left_turn.yaml``, baked into
each case as ``reference/artifacts/trajectory.yaml`` (+ ``action.npz``,
``actions.jsonl``). This tool writes each backend's native inputs for that turn
and the five-repeat queues the existing runners consume.

Both backends follow the recipes their prompt-consistency rollouts were
generated with, seeds 0-4:

* LingBot Base Cam: 161 camera poses at 16 fps -- one counter-clockwise turn
  in place in first person; two counter-clockwise orbits around the character,
  looking at it, in third person.
* Matrix-Game 3.0: a 177-frame key schedule for the deterministic adapter --
  ``j`` (turn left, upstream ``CAM_VALUE`` 0.1) every frame, plus ``d``
  (strafe right) in third person so the camera orbits the character, since
  Matrix-Game has no native orbit control.

Inputs and queues go to ``runs/prompt_scene_consistency/model_inputs/``; the
candidate manifest for scoring is ``runs/prompt_scene_consistency/candidates.jsonl``::

    python tools/generation/prepare_prompt_case_inputs.py
    python tools/generation/run_lingbot_table7.py \\
      --queue runs/prompt_scene_consistency/model_inputs/lingbot_world_cam/generation_queue.jsonl
    python tools/generation/run_matrix_game3.py \\
      --queue runs/prompt_scene_consistency/model_inputs/matrix_game_3/generation_queue.jsonl
    python tools/run_wmeval_lite.py run --suite lite \\
      --candidate-manifest runs/prompt_scene_consistency/candidates.jsonl \\
      --dimensions prompt_scene_consistency --output-dir reports/wmeval_lite/prompt_open_models

``bake`` (authoring only) copies the canonical trajectory and its compiled forms
into the case bundles; re-lock them with ``tools/dataset/build_lite_case.py`` afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import action_backends  # noqa: E402
from evaluation.lite.case_loader import LiteCase, load_case  # noqa: E402
from evaluation.lite.suite_loader import load_suite  # noqa: E402
from evaluation.trajectories import load_trajectory  # noqa: E402

SUITE_ROOT = ROOT / "lite"
DIMENSION = "prompt_scene_consistency"
TABLE_ROOT = ROOT / "experiments/protocol"
TRAJECTORY = "trajectories/prompt_camera_left_turn.yaml"
# Canonical compiled form -> name inside the case bundle.
BAKED = {
    TRAJECTORY: "trajectory.yaml",
    "compiled/prompt_camera_left_turn.action.npz": "action.npz",
    "compiled/prompt_camera_left_turn.actions.jsonl": "actions.jsonl",
}
DEFAULT_OUTPUT = ROOT / "runs/prompt_scene_consistency"
REPEATS = tuple(f"run{index}" for index in range(5))
# The prompt-consistency rollouts used seeds 0-4 for both models.
ROLLOUT_SEEDS = (0, 1, 2, 3, 4)
# LingBot Base Cam [fx, fy, cx, cy] at 480x832: the in-place turn's defaults and
# the upstream example intrinsics the orbit reused.
LINGBOT_TURN_INTRINSICS = np.array([415.5298, 415.6922, 415.77786, 239.77779], dtype=np.float32)
LINGBOT_ORBIT_INTRINSICS = np.array(
    [502.9115905761719, 503.1081237792969, 415.7778625488281, 239.7777862548828], dtype=np.float32
)
LINGBOT_ORBIT_TURNS = 2
# Matrix-Game 3.0 interactive keys (pipeline/inference_interactive_pipeline.py):
# j -> mouse [0, -CAM_VALUE], d -> keyboard [0, 0, 0, 1, 0, 0].
MG3_CAM_VALUE = 0.1
MG3_KEY_D = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_cases(suite_root: Path = SUITE_ROOT) -> list[LiteCase]:
    suite = load_suite(suite_root)
    return [
        load_case(suite.cases_root / case_id, allowed_cases=suite.all_case_ids())
        for case_id in suite.dimensions[DIMENSION].cases
    ]


def bake(cases: list[LiteCase]) -> list[str]:
    written = []
    for case in cases:
        for source, name in BAKED.items():
            target = case.artifacts_dir / name
            shutil.copyfile(TABLE_ROOT / source, target)
            written.append(str(target.relative_to(ROOT)))
    return written


def case_trajectory(case: LiteCase) -> dict[str, Any]:
    """The case's baked turn, which must be the canonical trajectory."""
    baked = case.artifacts_dir / "trajectory.yaml"
    if not baked.is_file() or sha256(baked) != sha256(TABLE_ROOT / TRAJECTORY):
        raise ValueError(f"{case.case_id}: reference/artifacts/trajectory.yaml is not {TRAJECTORY}")
    return load_trajectory(baked)


def lingbot_turn_in_place(frames: int) -> np.ndarray:
    """First person: one counter-clockwise 360 deg yaw sweep in place, the last
    pose stopping one step short of the start (build_poses.py defaults)."""
    sign = -1.0  # counter-clockwise on screen in the OpenCV c2w convention
    angles = np.deg2rad(0.0) + sign * np.linspace(
        0.0, np.deg2rad(360.0), frames, dtype=np.float32, endpoint=False
    )
    poses = np.zeros((frames, 4, 4), dtype=np.float32)
    poses[:, 3, 3] = 1.0
    for index, angle in enumerate(angles):
        c, s_ = float(np.cos(float(angle))), float(np.sin(float(angle)))
        rotation = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]], dtype=np.float32)
        # Composed onto the (identity) base pose exactly as the rollouts were.
        poses[index, :3, :3] = np.eye(3, dtype=np.float32) @ rotation
    return poses


def _look_at_opencv_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    z_axis = (target - eye) / np.linalg.norm(target - eye)
    x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x_axis, y_axis, z_axis, eye
    return c2w


def lingbot_orbit(frames: int) -> np.ndarray:
    """Third person: the camera orbits the character counter-clockwise, twice
    over the clip, looking at it (build_orbit_poses.py --turns 2
    --orbit_sign -1; radius 1.8, centre (0, 0, 2), eye height 0.12)."""
    center = np.array([0.0, 0.0, 2.0])
    angles = -np.linspace(0.0, 2.0 * np.pi * LINGBOT_ORBIT_TURNS, frames, endpoint=False)
    return np.stack(
        [
            _look_at_opencv_c2w(center + np.array([1.8 * np.sin(a), 0.12, 1.8 * np.cos(a)]), center)
            for a in angles
        ]
    )


def _relative(path: Path) -> str:
    """Repo-relative when possible (the runners resolve paths against the repo)."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def prepare_lingbot(case: LiteCase, trajectory: dict[str, Any], output_root: Path) -> dict[str, Any]:
    backend = action_backends.backend("lingbot_world_cam")
    action_backends.require_support("lingbot_world_cam", trajectory)
    frames, fps = int(backend["frames"]), float(backend["fps"])
    orbit = case.perspective == "third_person"
    poses = lingbot_orbit(frames) if orbit else lingbot_turn_in_place(frames)
    intrinsics = np.tile(LINGBOT_ORBIT_INTRINSICS if orbit else LINGBOT_TURN_INTRINSICS, (frames, 1))
    case_root = output_root / "model_inputs/lingbot_world_cam" / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    np.save(case_root / "poses.npy", poses)
    np.save(case_root / "intrinsics.npy", intrinsics)
    return {
        "model": "lingbot_world_cam",
        "frame_count": frames,
        "fps": fps,
        "mapping": {
            "kind": "orbit_around_character" if orbit else "turn_in_place",
            "pose_convention": "OpenCV camera-to-world",
            "turns": LINGBOT_ORBIT_TURNS if orbit else 1,
        },
        "seeds": list(ROLLOUT_SEEDS),
        "poses": _relative(case_root / "poses.npy"),
        "poses_sha256": sha256(case_root / "poses.npy"),
        "intrinsics": _relative(case_root / "intrinsics.npy"),
        "intrinsics_sha256": sha256(case_root / "intrinsics.npy"),
    }


def prepare_matrix_game_3(case: LiteCase, trajectory: dict[str, Any], output_root: Path) -> dict[str, Any]:
    backend = action_backends.backend("matrix_game_3")
    action_backends.require_support("matrix_game_3", trajectory)
    frames = int(backend["frames"])
    orbit = case.perspective == "third_person"
    keyboard = np.zeros((frames, int(backend["keyboard_width"])), dtype=np.float32)
    mouse = np.zeros((frames, 2), dtype=np.float32)
    mouse[:, 1] = -MG3_CAM_VALUE
    if orbit:
        keyboard[:, MG3_KEY_D] = 1.0
    mapping = {
        "kind": "interactive_key_recipe",
        "keys_per_frame": {"mouse": "j", "keyboard": "d" if orbit else "q"},
        "cam_value": MG3_CAM_VALUE,
        "upstream_geometric_degrees_per_frame": MG3_CAM_VALUE
        * float(backend["upstream_geometric_deg_per_mouse_unit"]),
    }
    case_root = output_root / "model_inputs/matrix_game_3" / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    schedule = case_root / "schedule.npz"
    np.savez(schedule, keyboard=keyboard, mouse=mouse)
    return {
        "model": "matrix_game_3",
        "frame_count": int(backend["frames"]),
        "fps": float(backend["fps"]),
        "mapping": mapping,
        "seeds": list(ROLLOUT_SEEDS),
        "schedule": _relative(schedule),
        "schedule_sha256": sha256(schedule),
    }


PREPARERS = {"lingbot_world_cam": prepare_lingbot, "matrix_game_3": prepare_matrix_game_3}


def prepare(cases: list[LiteCase], output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    candidates: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"schema_version": 1, "models": {}}
    for model, preparer in PREPARERS.items():
        queue_rows: list[dict[str, Any]] = []
        for case in cases:
            trajectory = case_trajectory(case)
            receipt = {
                "schema_version": 1,
                "case_id": case.case_id,
                **preparer(case, trajectory, output_root),
                "trajectory": _relative(case.artifacts_dir / "trajectory.yaml"),
                "trajectory_sha256": sha256(case.artifacts_dir / "trajectory.yaml"),
                "conditioning_image": _relative(case.first_frame),
                "conditioning_image_sha256": sha256(case.first_frame),
                "environment_prompt": case.prompt,
                "character_prompt": "",
            }
            receipt_path = output_root / "model_inputs" / model / case.case_id / "receipt.json"
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            for repeat in REPEATS:
                output = output_root / "model_outputs" / model / case.case_id / repeat / "candidate.mp4"
                row = {
                    key: receipt[key]
                    for key in ("model", "case_id", "conditioning_image", "environment_prompt", "character_prompt")
                    + (("poses", "intrinsics") if "poses" in receipt else ("schedule",))
                }
                if "seeds" in receipt:
                    row["seed"] = receipt["seeds"][REPEATS.index(repeat)]
                row.update(
                    {
                        "generation_repeat": repeat,
                        "input_receipt": _relative(receipt_path),
                        "candidate_video": _relative(output),
                    }
                )
                queue_rows.append(row)
                candidates.append(
                    {
                        "model": model,
                        "case_id": case.case_id,
                        "seed": repeat,
                        "candidate_video": str(output.relative_to(output_root)),
                    }
                )
        queue = output_root / "model_inputs" / model / "generation_queue.jsonl"
        queue.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in queue_rows), encoding="utf-8")
        summary["models"][model] = {
            "cases": len(cases),
            "generations": len(queue_rows),
            "queue": _relative(queue),
            "queue_sha256": sha256(queue),
        }
    manifest = output_root / "candidates.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates), encoding="utf-8")
    summary["candidate_manifest"] = _relative(manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", choices=("prepare", "bake"), default="prepare")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    cases = prompt_cases()
    if args.command == "bake":
        print(json.dumps(bake(cases), indent=2))
    else:
        print(json.dumps(prepare(cases, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
