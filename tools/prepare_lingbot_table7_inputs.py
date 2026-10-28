#!/usr/bin/env python3
"""Prepare fixed-side LingBot Base Cam inputs for all active Table 7 cells.

This tool does not run LingBot and does not claim response-side calibration. It
adapts the hashed reference pose/intrinsics timelines into the official Base
Cam array shapes and writes provenance receipts plus a five-repeat launch queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import action_backends  # noqa: E402
from evaluation.trajectories import integrated_total, load_trajectory  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "runs/tables_4_7/model_inputs/lingbot_world_cam"
BACKEND = action_backends.backend("lingbot_world_cam")
FRAME_COUNT = int(BACKEND["frames"])
FPS = float(BACKEND["fps"])
REPEATS = tuple(f"run{i}" for i in range(5))
TRAJECTORY_ROOT = REPO_ROOT / "experiments/tables_4_7/trajectories"


@dataclass(frozen=True)
class Plan:
    case_id: str
    source_poses: str
    source_intrinsics: str
    conditioning: str
    source_mode: str
    start_frame: int = 0
    target_yaw_deg: float | None = None


def _commanded_yaw_deg(variant: str) -> float:
    """Total yaw of the canonical camera trajectory, checked against this backend."""
    trajectory = load_trajectory(TRAJECTORY_ROOT / f"camera_{variant}.yaml")
    action_backends.require_support("lingbot_world_cam", trajectory)
    return integrated_total(trajectory, "yaw_rate_deg_s")


def _plans() -> tuple[Plan, ...]:
    plans: list[Plan] = []
    garden_root = "runs/unity/garden_bridge_gt/evaluation_artifacts"
    mirror_root = (
        "lite/cases/MirrorRollingBall/"
        "prompt_third/reference/artifacts"
    )
    # Reference-side anchoring (start frame on the periodic Unity orbit) is
    # per-case; the commanded yaw comes from the canonical semantic trajectory.
    camera_cases = tuple(
        (variant, start, _commanded_yaw_deg(variant))
        for variant, start in (("perturb_in", 0), ("perturb_out", 75), ("360_in", 0), ("360_out", 75))
    )
    for trajectory, start, yaw in camera_cases:
        orientation = "inward" if trajectory.endswith("in") else "outward"
        plans.append(
            Plan(
                f"table7/WanderingGarden/{trajectory}",
                f"{garden_root}/poses.npy",
                f"{garden_root}/intrinsics.npy",
                f"experiments/tables_4_7/conditioning_frames/garden_{orientation}.png",
                "yaw_select",
                start,
                yaw,
            )
        )
    for trajectory, _start, _yaw in camera_cases:
        capture = (
            "runs/tables_4_7/unity_scene_runs/captures/"
            f"table7__BilliardGame__{trajectory}"
        )
        plans.append(
            Plan(
                f"table7/BilliardGame/{trajectory}",
                f"{capture}/evaluation_artifacts/poses.npy",
                f"{capture}/evaluation_artifacts/intrinsics.npy",
                f"{capture}/frames/frame_0000.png",
                "timeline_resample",
            )
        )
    for trajectory, start, yaw in camera_cases:
        if trajectory.endswith("in"):
            conditioning = (
                "lite/cases/MirrorRollingBall/"
                "prompt_third/reference/ref.png"
            )
        else:
            conditioning = (
                "runs/tables_4_7/prepared_assets/cases/"
                f"table7__MirrorRollingBall__{trajectory}/conditioning.png"
            )
        plans.append(
            Plan(
                f"table7/MirrorRollingBall/{trajectory}",
                f"{mirror_root}/poses.npy",
                f"{mirror_root}/intrinsics_t4.npy",
                conditioning,
                "yaw_select",
                start,
                yaw,
            )
        )
    return tuple(plans)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_metadata(case_id: str) -> dict[str, str]:
    table_root = REPO_ROOT / "experiments/tables_4_7"
    manifest_path = table_root / "cases.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    case = next(
        (row for row in manifest["cases"] if str(row["case_id"]) == case_id),
        None,
    )
    if case is None or int(case["table"]) != 7:
        raise ValueError(f"missing active Table 7 case metadata: {case_id}")
    components_path = table_root / str(manifest["prompt_components"])
    components = yaml.safe_load(components_path.read_text(encoding="utf-8"))
    variant = case_id.rsplit("/", 1)[-1]
    motion = "perturb" if variant.startswith("perturb_") else "360"
    prompt = " ".join(
        str(part).strip()
        for part in (
            components["scenes"][case["scene"]],
            components["orientations"][case["paper_orientation"]],
            components["motions"][motion],
        )
    )
    return {
        "environment_prompt": prompt,
        "character_prompt": str(case.get("character_prompt", "")),
        "prompt_provenance_status": str(
            manifest.get("prompt_provenance", {}).get("status", "missing")
        ),
        "cases_manifest_sha256": sha256(manifest_path),
        "prompt_components_sha256": sha256(components_path),
    }


def yaw_offsets(case_id: str, target_yaw_deg: float) -> np.ndarray:
    # 161 frames at 16 fps span exactly 10 seconds, including both endpoints.
    fraction = np.arange(FRAME_COUNT, dtype=np.float64) / (FRAME_COUNT - 1)
    if "/perturb_" in case_id:
        # Canonical trajectory turns for 1.25 s and holds for the remainder of
        # the 10 s horizon.
        fraction = np.clip(fraction * 10.0 / 1.25, 0.0, 1.0)
    return fraction * target_yaw_deg


def pose_yaw_source_indices(
    poses: np.ndarray, *, start_frame: int, offsets_deg: np.ndarray
) -> np.ndarray:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < 2:
        raise ValueError(f"poses must have shape (T, 4, 4), got {poses.shape}")
    if not 0 <= start_frame < poses.shape[0]:
        raise ValueError(f"start frame {start_frame} is outside pose timeline")
    forward = poses[:, :3, 2].astype(np.float64)
    horizontal_norm = np.linalg.norm(forward[:, [0, 2]], axis=1)
    if np.any(horizontal_norm < 1e-6):
        raise ValueError("camera pose timeline contains a vertical forward vector")
    yaw = np.rad2deg(np.arctan2(forward[:, 0], forward[:, 2]))
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(yaw)))
    rate = float(np.median(np.diff(unwrapped)))
    if abs(rate) < 1e-6:
        raise ValueError("camera pose timeline has no usable yaw motion")
    orbit_frames = abs(360.0 / rate)
    frame_numbers = np.arange(poses.shape[0])
    selected: list[int] = []
    for offset in np.asarray(offsets_deg, dtype=np.float64):
        target = yaw[start_frame] + offset
        expected = start_frame + offset / rate
        error = np.abs((yaw - target + 180.0) % 360.0 - 180.0)
        tied = frame_numbers[error <= float(error.min()) + 0.05]
        while expected < 0:
            expected += orbit_frames
        while expected >= poses.shape[0]:
            expected -= orbit_frames
        selected.append(int(tied[np.argmin(np.abs(tied - expected))]))
    return np.asarray(selected, dtype=np.int64)


def proper_opencv_c2w(poses: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert left-handed Unity-world C2W matrices to proper rotations.

    LingBot interpolates rotations with scipy, which correctly rejects the
    determinant -1 matrices emitted by the Unity captures.  Reflecting the
    world Z axis changes only the world basis: relative camera transforms and
    therefore the requested path are preserved.
    """
    determinants = np.linalg.det(poses[:, :3, :3])
    if np.all(determinants > 0.0):
        return poses.copy(), "already_right_handed"
    if not np.all(determinants < 0.0):
        raise ValueError("pose timeline mixes proper and reflected rotations")
    reflection = np.diag([1.0, 1.0, -1.0, 1.0]).astype(poses.dtype)
    converted = reflection[None] @ poses
    if not np.all(np.linalg.det(converted[:, :3, :3]) > 0.0):
        raise ValueError("failed to convert Unity poses to proper rotations")
    return converted, "unity_world_z_reflection_to_right_handed"


def _validate_sources(plan: Plan) -> tuple[Path, Path, Path, np.ndarray, np.ndarray]:
    pose_path = REPO_ROOT / plan.source_poses
    intrinsics_path = REPO_ROOT / plan.source_intrinsics
    conditioning_path = REPO_ROOT / plan.conditioning
    missing = [
        str(path)
        for path in (pose_path, intrinsics_path, conditioning_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing packaged input(s): " + ", ".join(missing))
    poses = np.load(pose_path, allow_pickle=False)
    intrinsics = np.load(intrinsics_path, allow_pickle=False)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"invalid poses shape for {plan.case_id}: {poses.shape}")
    if intrinsics.shape != (poses.shape[0], 4):
        raise ValueError(
            f"invalid intrinsics shape for {plan.case_id}: {intrinsics.shape}; "
            f"expected ({poses.shape[0]}, 4)"
        )
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError(f"non-finite camera values for {plan.case_id}")
    return pose_path, intrinsics_path, conditioning_path, poses, intrinsics


def prepare(plan: Plan, output_root: Path) -> dict[str, object]:
    pose_path, intrinsics_path, conditioning_path, poses, intrinsics = _validate_sources(plan)
    prompt = prompt_metadata(plan.case_id)
    if plan.source_mode == "yaw_select":
        if plan.target_yaw_deg is None:
            raise ValueError(f"missing target yaw for {plan.case_id}")
        offsets = yaw_offsets(plan.case_id, plan.target_yaw_deg)
        indices = pose_yaw_source_indices(
            poses, start_frame=plan.start_frame, offsets_deg=offsets
        )
    elif plan.source_mode == "timeline_resample":
        indices = np.rint(np.linspace(0, poses.shape[0] - 1, FRAME_COUNT)).astype(np.int64)
        offsets = None
    else:
        raise ValueError(f"unsupported source mode: {plan.source_mode}")

    output_dir = output_root / plan.case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_poses = output_dir / "poses.npy"
    output_intrinsics = output_dir / "intrinsics.npy"
    selected_poses, handedness_conversion = proper_opencv_c2w(poses[indices])
    np.save(output_poses, selected_poses)
    np.save(output_intrinsics, intrinsics[indices])

    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "prepared_requires_backend_validation",
        "claim": "fixed_side_input_only_not_model_calibration",
        "model": "robbyant/lingbot-world-base-cam",
        "model_revision": None,
        "case_id": plan.case_id,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "duration_sec": (FRAME_COUNT - 1) / FPS,
        "pose_convention": "OpenCV camera-to-world in a right-handed world basis",
        "pose_handedness_conversion": handedness_conversion,
        "intrinsics_convention": "fx_fy_cx_cy",
        "source_mode": plan.source_mode,
        "source_start_frame": plan.start_frame,
        "target_yaw_deg": plan.target_yaw_deg,
        "source_indices": indices.tolist(),
        "source_poses": plan.source_poses,
        "source_poses_sha256": sha256(pose_path),
        "source_intrinsics": plan.source_intrinsics,
        "source_intrinsics_sha256": sha256(intrinsics_path),
        "conditioning_image": plan.conditioning,
        "conditioning_image_sha256": sha256(conditioning_path),
        **prompt,
        "poses": str(output_poses.relative_to(REPO_ROOT)),
        "poses_sha256": sha256(output_poses),
        "intrinsics": str(output_intrinsics.relative_to(REPO_ROOT)),
        "intrinsics_sha256": sha256(output_intrinsics),
        "source_plan": asdict(plan),
    }
    if offsets is not None:
        receipt["requested_yaw_offsets_deg"] = offsets.tolist()
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    receipt["receipt"] = str(receipt_path.relative_to(REPO_ROOT))
    receipt["receipt_sha256"] = sha256(receipt_path)
    return receipt


def run(output_root: Path) -> dict[str, object]:
    receipts = [prepare(plan, output_root) for plan in _plans()]
    queue_path = output_root / "generation_queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as handle:
        for receipt in receipts:
            for repeat in REPEATS:
                item = {
                    "status": "waiting_for_checkpoint_and_gpu_window",
                    "model": "lingbot_world_cam",
                    "case_id": receipt["case_id"],
                    "generation_repeat": repeat,
                    "conditioning_image": receipt["conditioning_image"],
                    "environment_prompt": receipt["environment_prompt"],
                    "character_prompt": receipt["character_prompt"],
                    "prompt_provenance_status": receipt["prompt_provenance_status"],
                    "poses": receipt["poses"],
                    "intrinsics": receipt["intrinsics"],
                    "input_receipt": receipt["receipt"],
                    "candidate_video": (
                        f"runs/tables_4_7/model_outputs/lingbot_world_cam/"
                        f"{receipt['case_id']}/{repeat}/candidate.mp4"
                    ),
                }
                handle.write(json.dumps(item, sort_keys=True) + "\n")
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "fixed_side_inputs_prepared_runtime_calibration_pending",
        "case_count": len(receipts),
        "generation_count": len(receipts) * len(REPEATS),
        "frame_count_per_case": FRAME_COUNT,
        "fps": FPS,
        "queue": str(queue_path.relative_to(REPO_ROOT)),
        "queue_sha256": sha256(queue_path),
        "cases": [
            {
                key: receipt[key]
                for key in (
                    "case_id",
                    "poses",
                    "poses_sha256",
                    "intrinsics",
                    "intrinsics_sha256",
                    "receipt",
                    "receipt_sha256",
                )
            }
            for receipt in receipts
        ],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    summary = run(output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
