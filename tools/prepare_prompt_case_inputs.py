#!/usr/bin/env python3
"""Prepare LingBot (Cam) and Matrix-Game 3.0 inputs for the prompt-scene cases.

Every ``prompt_scene_consistency`` case turns the camera left for 10 s at the
Unity reference's 40 deg/s: the canonical semantic trajectory
``experiments/tables_4_7/trajectories/prompt_camera_left_turn.yaml``, baked into
each case as ``reference/artifacts/trajectory.yaml`` (+ ``action.npz``,
``actions.jsonl``). This tool compiles that turn onto each backend's native
controls and writes the five-repeat queues the existing runners consume:

* LingBot Base Cam: the case's Unity camera poses (in place in first person,
  around the character in third person) resampled to 161 frames at 16 fps,
  with per-frame ``[fx, fy, cx, cy]`` from the case intrinsics.
* Matrix-Game 3.0: a 177-frame mouse-yaw schedule using the registry's
  response-calibrated full-turn gain (camera yaw only, as for Table 7).

Inputs and queues go to ``runs/prompt_scene_consistency/model_inputs/``; the
candidate manifest for scoring is ``runs/prompt_scene_consistency/candidates.jsonl``::

    python tools/prepare_prompt_case_inputs.py
    python tools/run_lingbot_table7.py \\
      --queue runs/prompt_scene_consistency/model_inputs/lingbot_world_cam/generation_queue.jsonl
    python tools/run_matrix_game3.py \\
      --queue runs/prompt_scene_consistency/model_inputs/matrix_game_3/generation_queue.jsonl
    python tools/run_wmeval_lite.py run --suite eval_sets/wmeval_lite_v1 \\
      --candidate-manifest runs/prompt_scene_consistency/candidates.jsonl \\
      --dimensions prompt_scene_consistency --output-dir reports/wmeval_lite/prompt_open_models

``bake`` (authoring only) copies the canonical trajectory and its compiled forms
into the case bundles; re-lock them with ``tools/build_lite_case.py`` afterwards.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import action_backends  # noqa: E402
from evaluation.lite.case_loader import LiteCase, load_case  # noqa: E402
from evaluation.lite.suite_loader import load_suite  # noqa: E402
from evaluation.trajectories import integrated_total, load_trajectory  # noqa: E402
from tools.prepare_lingbot_table7_inputs import proper_opencv_c2w  # noqa: E402
from tools.prepare_matrix_game_inputs import schedule_for  # noqa: E402

SUITE_ROOT = ROOT / "eval_sets/wmeval_lite_v1"
DIMENSION = "prompt_scene_consistency"
TABLE_ROOT = ROOT / "experiments/tables_4_7"
TRAJECTORY = "trajectories/prompt_camera_left_turn.yaml"
# Canonical compiled form -> name inside the case bundle.
BAKED = {
    TRAJECTORY: "trajectory.yaml",
    "compiled/prompt_camera_left_turn.action.npz": "action.npz",
    "compiled/prompt_camera_left_turn.actions.jsonl": "actions.jsonl",
}
DEFAULT_OUTPUT = ROOT / "runs/prompt_scene_consistency"
REPEATS = tuple(f"run{index}" for index in range(5))
UNITY_FPS = 30.0
MAX_RATE_ERROR_DEG_S = 1.0


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


def unity_camera(case: LiteCase, trajectory: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Unity poses and (T, 4) intrinsics, checked to perform the commanded turn."""
    poses = np.load(case.artifacts_dir / "poses.npy", allow_pickle=False)
    k = np.load(case.artifacts_dir / "intrinsics.npy", allow_pickle=False)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or k.shape != (3, 3):
        raise ValueError(f"{case.case_id}: unexpected poses {poses.shape} / intrinsics {k.shape}")
    forward = poses[:, :3, 2].astype(np.float64)
    yaw = np.rad2deg(np.unwrap(np.arctan2(forward[:, 0], forward[:, 2])))
    rate = float(np.median(np.diff(yaw))) * UNITY_FPS
    commanded = integrated_total(trajectory, "yaw_rate_deg_s") / float(trajectory["duration_sec"])
    if abs(rate - commanded) > MAX_RATE_ERROR_DEG_S:
        raise ValueError(f"{case.case_id}: Unity turns at {rate:.2f} deg/s, trajectory commands {commanded}")
    intrinsics = np.tile([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], (poses.shape[0], 1)).astype(np.float32)
    return poses, intrinsics


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
    poses, intrinsics = unity_camera(case, trajectory)
    # Same timeline resampling as the Table 7 Unity captures: the reference
    # spans the commanded 10 s, so uniform indices follow the commanded turn.
    indices = np.rint(np.linspace(0, poses.shape[0] - 1, frames)).astype(np.int64)
    selected, handedness = proper_opencv_c2w(poses[indices])
    case_root = output_root / "model_inputs/lingbot_world_cam" / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    np.save(case_root / "poses.npy", selected)
    np.save(case_root / "intrinsics.npy", intrinsics[indices])
    return {
        "model": "lingbot_world_cam",
        "frame_count": frames,
        "fps": fps,
        "mapping": {
            "kind": "unity_camera_poses",
            "source_mode": "timeline_resample",
            "source_indices": indices.tolist(),
            "pose_handedness_conversion": handedness,
            "target_yaw_deg": integrated_total(trajectory, "yaw_rate_deg_s"),
        },
        "poses": _relative(case_root / "poses.npy"),
        "poses_sha256": sha256(case_root / "poses.npy"),
        "intrinsics": _relative(case_root / "intrinsics.npy"),
        "intrinsics_sha256": sha256(case_root / "intrinsics.npy"),
        "source_poses": _relative(case.artifacts_dir / "poses.npy"),
        "source_poses_sha256": sha256(case.artifacts_dir / "poses.npy"),
    }


def prepare_matrix_game_3(case: LiteCase, trajectory: dict[str, Any], output_root: Path) -> dict[str, Any]:
    backend = action_backends.backend("matrix_game_3")
    keyboard, mouse, mapping = schedule_for(
        "matrix_game_3", {"case_id": case.case_id, "trajectory": TRAJECTORY}
    )
    case_root = output_root / "model_inputs/matrix_game_3" / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    schedule = case_root / "schedule.npz"
    np.savez(schedule, keyboard=keyboard, mouse=mouse)
    return {
        "model": "matrix_game_3",
        "frame_count": int(backend["frames"]),
        "fps": float(backend["fps"]),
        "mapping": mapping,
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
