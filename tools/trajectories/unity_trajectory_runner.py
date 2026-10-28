#!/usr/bin/env python3
"""Probe and calibrate the local WanderingGarden ML-Agents build.

This intentionally lives in the isolated ``.venv-unity`` environment because
ML-Agents 1.1.0 requires NumPy/protobuf versions incompatible with WMEval's
main evaluation environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = (
    ROOT.parent
    / "WorldModelBenchScenePrivate/GameScenes/WanderingGarden/Builds/WanderingGarden/macOS"
    / "WanderingGarden.app"
)
GARDEN_BRIDGE_CAMERA_POSITION = (3.277, 1.600, 69.049)
GARDEN_INWARD_YAW_DEG = 298.2


def _heading_deg(vector: np.ndarray) -> float:
    """Unity heading in degrees from a [x, y, z] forward vector."""
    return math.degrees(math.atan2(float(vector[0]), float(vector[2])))


def _angle_delta_deg(start: float, end: float) -> float:
    return (end - start + 180.0) % 360.0 - 180.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_spec(spec: Any) -> dict[str, Any]:
    return {
        "observation_shapes": [list(item.shape) for item in spec.observation_specs],
        "continuous_actions": int(spec.action_spec.continuous_size),
        "discrete_branches": [int(value) for value in spec.action_spec.discrete_branches],
    }


def _vector_observation(decision_steps: Any) -> np.ndarray:
    for observation in decision_steps.obs:
        if observation.ndim == 2 and observation.shape[1] == 50:
            return observation
    raise RuntimeError("GardenWalker 50-float vector observation was not returned")


def _save_first_visual_observation(decision_steps: Any, path: Path) -> bool:
    from PIL import Image

    for observation in decision_steps.obs:
        if observation.ndim != 4 or observation.shape[-1] not in (1, 3, 4):
            continue
        image = np.clip(observation[0] * 255.0, 0, 255).astype(np.uint8)
        if image.shape[-1] == 1:
            image = image[..., 0]
        Image.fromarray(image).save(path)
        return True
    return False


def _open_environment(build: Path, seed: int, worker_id: int):
    try:
        from mlagents_envs.environment import UnityEnvironment
    except ImportError as exc:
        raise SystemExit(
            "ML-Agents is unavailable. Use .venv-unity/bin/python and "
            "requirements/unity-macos.lock."
        ) from exc
    if not build.exists():
        raise SystemExit(f"Unity application not found: {build}")
    return UnityEnvironment(
        file_name=str(build),
        worker_id=worker_id,
        seed=seed,
        no_graphics=False,
        timeout_wait=60,
        additional_args=[
            "-wmeval-mlagents",
            "-screen-width",
            "640",
            "-screen-height",
            "480",
        ],
        log_folder=str(ROOT / "runs/unity/logs"),
    )


def _single_behavior(env: Any) -> str:
    names = list(env.behavior_specs)
    garden = [name for name in names if name.startswith("GardenWalker")]
    if len(garden) != 1:
        raise RuntimeError(f"expected one GardenWalker behavior, got {names}")
    return garden[0]


def _set_action(env: Any, behavior: str, count: int, values: Sequence[float]) -> None:
    from mlagents_envs.base_env import ActionTuple

    continuous = np.tile(np.asarray(values, dtype=np.float32), (count, 1))
    env.set_actions(behavior, ActionTuple(continuous=continuous))


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = _open_environment(args.build.resolve(), args.seed, args.worker_id)
    try:
        env.reset()
        behavior = _single_behavior(env)
        decisions, terminals = env.get_steps(behavior)
        if len(decisions) == 0:
            raise RuntimeError("Unity returned no GardenWalker decision agent after reset")
        vector = _vector_observation(decisions)
        first_frame = output / "unity_frame0.png"
        saved = _save_first_visual_observation(decisions, first_frame)
        report = {
            "status": "ok",
            "build": str(args.build.resolve()),
            "behavior": behavior,
            "spec": _jsonable_spec(env.behavior_specs[behavior]),
            "decision_agents": int(len(decisions)),
            "terminal_agents": int(len(terminals)),
            "initial_position_xyz": vector[0, 0:3].astype(float).tolist(),
            "initial_forward_xyz": vector[0, 3:6].astype(float).tolist(),
            "initial_heading_deg": _heading_deg(vector[0, 3:6]),
            "first_frame": first_frame.name if saved else None,
        }
        _set_action(env, behavior, len(decisions), [0.0, 0.0, 0.0, 0.0])
        env.step()
        next_decisions, _ = env.get_steps(behavior)
        report["post_step_decision_agents"] = int(len(next_decisions))
    finally:
        env.close()
    (output / "probe.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = _open_environment(args.build.resolve(), args.seed, args.worker_id)
    rows: list[dict[str, Any]] = []
    try:
        for level in args.levels:
            env.reset()
            behavior = _single_behavior(env)
            decisions, _ = env.get_steps(behavior)
            before = _vector_observation(decisions)[0]
            start_heading = _heading_deg(before[3:6])
            for _ in range(args.decisions):
                decisions, _ = env.get_steps(behavior)
                if len(decisions) == 0:
                    raise RuntimeError("GardenWalker stopped requesting decisions")
                _set_action(env, behavior, len(decisions), [0.0, 0.0, level, 0.0])
                env.step()
            decisions, _ = env.get_steps(behavior)
            after = _vector_observation(decisions)[0]
            end_heading = _heading_deg(after[3:6])
            duration = args.decisions * args.decision_period * args.fixed_timestep_sec
            delta = _angle_delta_deg(start_heading, end_heading)
            rows.append(
                {
                    "look_x": level,
                    "decisions": args.decisions,
                    "duration_sec": duration,
                    "start_heading_deg": start_heading,
                    "end_heading_deg": end_heading,
                    "signed_delta_deg": delta,
                    "yaw_rate_deg_sec": delta / duration,
                }
            )
    finally:
        env.close()
    usable = [row for row in rows if abs(row["look_x"]) > 1e-9]
    gains = [row["yaw_rate_deg_sec"] / row["look_x"] for row in usable]
    report = {
        "status": "ok",
        "build": str(args.build.resolve()),
        "fixed_timestep_sec": args.fixed_timestep_sec,
        "decision_period": args.decision_period,
        "samples": rows,
        "median_yaw_rate_deg_sec_per_unit": float(np.median(gains)),
    }
    (output / "yaw_calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _player_binary(application: Path) -> Path:
    if application.suffix != ".app":
        return application
    preferred = application / "Contents" / "MacOS" / application.stem
    if preferred.is_file():
        return preferred
    candidates = [path for path in (application / "Contents" / "MacOS").iterdir() if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"could not identify player executable in {application}")
    return candidates[0]


def _validate_ground_truth(capture_dir: Path, capture_fps: float) -> dict[str, Any]:
    data_path = capture_dir / "captured_data.json"
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    frames = raw.get("frames", [])
    count = int(raw.get("num_frames", -1))
    if count != len(frames) or count < 2:
        raise RuntimeError(f"invalid ground-truth frame count: declared {count}, found {len(frames)}")
    frame_paths = [capture_dir / "frames" / f"frame_{index:04d}.png" for index in range(count)]
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} rendered ground-truth frames")
    intrinsics = np.asarray(
        [[row["fx"], row["fy"], row["cx"], row["cy"]] for row in frames],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(intrinsics)) or np.any(intrinsics[:, :2] <= 0):
        raise RuntimeError("capture contains invalid camera intrinsics")
    if not np.allclose(intrinsics[:, 0], intrinsics[:, 1], rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            "Unity RenderTexture capture requires square-pixel intrinsics (fx == fy)"
        )
    positions = np.asarray([row["cam_pos"] for row in frames], dtype=np.float64)
    headings = np.asarray(
        [math.atan2(float(row["c2w"][2]), float(row["c2w"][10])) for row in frames],
        dtype=np.float64,
    )
    unwrapped_deg = np.degrees(np.unwrap(headings))
    deltas = np.diff(unwrapped_deg)
    rate = float(np.median(deltas) * capture_fps)
    position_drift = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
    represented_sweep = float(unwrapped_deg[-1] - unwrapped_deg[0])
    inferred_full_sweep = represented_sweep + float(np.median(deltas))
    if position_drift > 1e-4:
        raise RuntimeError(f"camera translation drifted by {position_drift} m")
    if abs(abs(inferred_full_sweep) - 360.0) > 0.1:
        raise RuntimeError(f"capture is not a 360-degree sweep: {inferred_full_sweep}")
    first_frame = frame_paths[0]
    opposite_frame = frame_paths[count // 2]
    return {
        "status": "ok",
        "capture_dir": str(capture_dir),
        "frame_count": count,
        "resolution": raw.get("resolution"),
        "capture_fps": capture_fps,
        "duration_sec": count / capture_fps,
        "represented_sweep_deg": represented_sweep,
        "inferred_full_sweep_deg": inferred_full_sweep,
        "median_yaw_rate_deg_sec": rate,
        "max_camera_position_drift_m": position_drift,
        "conditioning_frame": str(first_frame),
        "conditioning_frame_sha256": _sha256(first_frame),
        "orientation_candidates": {
            "inward_initial": {
                "frame": str(first_frame),
                "sha256": _sha256(first_frame),
                "yaw_offset_deg": 0.0,
            },
            "outward_opposite_yaw": {
                "frame": str(opposite_frame),
                "sha256": _sha256(opposite_frame),
                "yaw_offset_deg": float(unwrapped_deg[count // 2] - unwrapped_deg[0]),
                "status": "candidate_requires_paper_orientation_confirmation",
            },
        },
        "camera_log_sha256": _sha256(data_path),
    }


def _export_evaluation_artifacts(
    capture_dir: Path, capture_fps: float
) -> dict[str, Any]:
    """Convert the capture log into environment-object evaluator artifacts."""
    raw = json.loads((capture_dir / "captured_data.json").read_text(encoding="utf-8"))
    frames = raw.get("frames", [])
    anchors = raw.get("anchors", {})
    count = int(raw.get("num_frames", -1))
    if count != len(frames) or count < 2 or not isinstance(anchors, dict):
        raise RuntimeError("capture log cannot be exported for evaluation")
    artifact_dir = capture_dir / "evaluation_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    poses = np.asarray(
        [np.asarray(row["c2w"], dtype=np.float64).reshape(4, 4) for row in frames],
        dtype=np.float64,
    )
    intrinsics = np.asarray(
        [[row["fx"], row["fy"], row["cx"], row["cy"]] for row in frames],
        dtype=np.float64,
    )
    if not np.allclose(intrinsics[:, 0], intrinsics[:, 1], rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            "Unity RenderTexture capture requires square-pixel intrinsics (fx == fy)"
        )
    poses_path = artifact_dir / "poses.npy"
    intrinsics_path = artifact_dir / "intrinsics.npy"
    np.save(poses_path, poses, allow_pickle=False)
    np.save(intrinsics_path, intrinsics, allow_pickle=False)

    indices = list(range(count))
    objects: dict[str, Any] = {}
    for label, position in anchors.items():
        point = [float(value) for value in position]
        objects[str(label)] = {
            "frames": indices,
            "positions_world": [point for _ in indices],
        }
    if "Main tree" in objects and "tree" not in objects:
        objects["tree"] = objects["Main tree"]
    transforms_path = artifact_dir / "object_transforms.json"
    transforms_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": str((capture_dir / "captured_data.json").resolve()),
                "objects": objects,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the Unity reference video")
    reference_video = artifact_dir / "reference.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(capture_fps),
            "-start_number",
            "0",
            "-i",
            str(capture_dir / "frames" / "frame_%04d.png"),
            "-frames:v",
            str(count),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(reference_video),
        ],
        check=True,
    )
    artifacts = {
        "reference_video": reference_video,
        "object_transforms_json": transforms_path,
        "poses_npy": poses_path,
        "intrinsics_npy": intrinsics_path,
    }
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in artifacts.items()
    }


def run_ground_truth_capture(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    data_path = output / "captured_data.json"
    if not (args.reuse and data_path.is_file()):
        output.mkdir(parents=True, exist_ok=True)
        binary = _player_binary(args.build.resolve())
        command = [
            str(binary),
            "-outDir",
            str(output),
            "-screen-width",
            "1280",
            "-screen-height",
            "720",
            "-wmeval-cam-pos",
            ",".join(f"{value:.6f}" for value in args.camera_position),
            "-wmeval-cam-yaw",
            f"{args.camera_yaw:.6f}",
        ]
        subprocess.run(command, check=True)
    report = _validate_ground_truth(output, args.capture_fps)
    expected_position = np.asarray(args.camera_position, dtype=np.float64)
    captured = json.loads((output / "captured_data.json").read_text(encoding="utf-8"))
    actual_position = np.asarray(captured["frames"][0]["cam_pos"], dtype=np.float64)
    pose_error = float(np.linalg.norm(actual_position - expected_position))
    if pose_error > 1e-4:
        raise RuntimeError(
            "capture did not honor the requested in-garden camera position: "
            f"requested={expected_position.tolist()} actual={actual_position.tolist()}"
        )
    report["build"] = str(args.build.resolve())
    report["requested_camera_position_xyz"] = expected_position.tolist()
    report["requested_camera_yaw_deg"] = float(args.camera_yaw)
    report["camera_position_error_m"] = pose_error
    report["spawn_semantics"] = "camera positioned on GardenBridge, facing the main cherry tree"
    report["evaluation_artifacts"] = _export_evaluation_artifacts(
        output, args.capture_fps
    )
    (output / "capture_receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/unity/garden")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe")
    calibrate = subparsers.add_parser("calibrate-yaw")
    calibrate.add_argument("--levels", type=float, nargs="+", default=[-1.0, -0.5, 0.5, 1.0])
    calibrate.add_argument("--decisions", type=int, default=10)
    calibrate.add_argument("--decision-period", type=int, default=5)
    calibrate.add_argument("--fixed-timestep-sec", type=float, default=0.02)
    capture = subparsers.add_parser("capture-ground-truth")
    capture.add_argument("--capture-fps", type=float, default=15.0)
    capture.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=GARDEN_BRIDGE_CAMERA_POSITION,
        help="Unity world-space camera spawn; defaults to the GardenBridge",
    )
    capture.add_argument(
        "--camera-yaw",
        type=float,
        default=GARDEN_INWARD_YAW_DEG,
        help="initial Unity yaw; default faces the main cherry tree from the GardenBridge",
    )
    capture.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "probe":
        report = run_probe(args)
    elif args.command == "calibrate-yaw":
        report = run_calibration(args)
    else:
        report = run_ground_truth_capture(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
