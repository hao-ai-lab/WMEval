#!/usr/bin/env python3
"""Standardize repeated model runs for an existing WMEval Lite case."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.provenance_paths import resolve_recorded_path

REPEATS = tuple(f"run{index}" for index in range(5))


def media_tool(name: str) -> str | None:
    """Resolve a system media binary or the repository-pinned fallback."""
    system = shutil.which(name)
    if system:
        return system
    bundled = REPO_ROOT / ".local" / "bin" / name
    if bundled.is_file():
        return str(bundled)
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    try:
        result = {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()
    if (
        result["frame_count"] < 2
        or result["fps"] <= 0
        or result["width"] <= 0
        or result["height"] <= 0
    ):
        raise ValueError(f"invalid video metadata: {path}")
    return result


def verified_artifact(
    receipt: dict[str, Any], *, receipt_dir: Path = Path(".")
) -> tuple[Path, str]:
    status = receipt.get("evaluation_artifact_status")
    if status == "native_action_aligned_ready":
        path_key, hash_key, kind = "evaluation_video", "evaluation_video_sha256", "native_aligned"
    elif status == "action_aligned_capture_ready":
        path_key, hash_key, kind = "observed_video", "observed_video_sha256", "browser_capture"
    else:
        raise ValueError(f"unsupported evaluation artifact status: {status!r}")
    path = resolve_recorded_path(
        str(receipt.get(path_key, "")),
        repo_root=REPO_ROOT,
        relative_to=receipt_dir,
    )
    expected = receipt.get(hash_key)
    if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
        raise ValueError(f"missing or hash-mismatched {path_key}: {path}")
    return path, kind


def verified_browser_capture(
    receipt: dict[str, Any], *, receipt_dir: Path = Path(".")
) -> tuple[Path, str]:
    """Return the hashed browser capture even when a native export also exists."""
    path = resolve_recorded_path(
        str(receipt.get("observed_video", "")),
        repo_root=REPO_ROOT,
        relative_to=receipt_dir,
    )
    expected = receipt.get("observed_video_sha256")
    if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
        raise ValueError(f"missing or hash-mismatched observed_video: {path}")
    return path, "browser_capture"


def select_common_artifacts(
    receipts: list[dict[str, Any]], receipt_dirs: list[Path] | None = None
) -> tuple[list[tuple[Path, str]], str]:
    """Use one render pipeline for every repeat in an aggregate.

    Native exports and browser captures have different crops, resolutions, and
    sometimes different camera framing.  Mixing them creates artificial
    between-repeat variance.  Prefer native-aligned videos only when every
    repeat has one; otherwise use each repeat's independently hashed browser
    capture.
    """
    if receipt_dirs is None:
        receipt_dirs = [Path(".")] * len(receipts)
    all_native = all(
        receipt.get("evaluation_artifact_status") == "native_action_aligned_ready"
        for receipt in receipts
    )
    if all_native:
        return [
            verified_artifact(receipt, receipt_dir=receipt_dir)
            for receipt, receipt_dir in zip(receipts, receipt_dirs, strict=True)
        ], "all_native_aligned"
    return (
        [
            verified_browser_capture(receipt, receipt_dir=receipt_dir)
            for receipt, receipt_dir in zip(receipts, receipt_dirs, strict=True)
        ],
        "all_browser_capture_due_to_incomplete_native_coverage",
    )


def pose_yaw_source_indices(
    poses: np.ndarray,
    *,
    start_frame: int,
    yaw_offsets_deg: np.ndarray,
) -> list[int]:
    """Select reference poses nearest to requested yaw, including orbit wrap.

    This uses camera-forward vectors instead of assuming one frame equals a
    fixed angle. It therefore works for the MirrorRollingBall reference, whose
    300-frame ArrowLeft recording covers about 399 degrees rather than exactly
    one turn.
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < 2:
        raise ValueError(f"poses must have shape (T, 4, 4), got {poses.shape}")
    if not 0 <= start_frame < poses.shape[0]:
        raise ValueError(f"start frame {start_frame} is outside the pose timeline")
    forward = poses[:, :3, 2].astype(np.float64)
    horizontal_norm = np.linalg.norm(forward[:, [0, 2]], axis=1)
    if np.any(horizontal_norm < 1e-6):
        raise ValueError("camera pose timeline contains a vertical forward vector")
    yaw = np.rad2deg(np.arctan2(forward[:, 0], forward[:, 2]))
    unwrapped_yaw = np.rad2deg(np.unwrap(np.deg2rad(yaw)))
    yaw_rate_per_frame = float(np.median(np.diff(unwrapped_yaw)))
    if abs(yaw_rate_per_frame) < 1e-6:
        raise ValueError("camera pose timeline has no usable yaw motion")
    orbit_frames = abs(360.0 / yaw_rate_per_frame)
    requested_offsets = np.asarray(yaw_offsets_deg, dtype=np.float64)
    requested = yaw[start_frame] + requested_offsets
    expected_frames = start_frame + requested_offsets / yaw_rate_per_frame
    indices: list[int] = []
    frame_numbers = np.arange(poses.shape[0])
    for target, expected_frame in zip(requested, expected_frames):
        error = np.abs((yaw - target + 180.0) % 360.0 - 180.0)
        minimum = float(error.min())
        tied = frame_numbers[error <= minimum + 0.05]
        # A >360-degree source has duplicated orientations. Follow the
        # requested continuous direction, wrapping only when that conceptual
        # frame falls outside the recorded orbit.
        canonical_frame = float(expected_frame)
        while canonical_frame < 0:
            canonical_frame += orbit_frames
        while canonical_frame >= poses.shape[0]:
            canonical_frame -= orbit_frames
        selected = int(tied[np.argmin(np.abs(tied - canonical_frame))])
        indices.append(selected)
    return indices


def _yaw_offsets_for_run(
    *,
    receipt: dict[str, Any],
    output_count: int,
    trajectory_mode: str,
    target_yaw_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if trajectory_mode == "full_turn":
        offsets = np.arange(output_count, dtype=np.float64) / output_count * target_yaw_deg
        return offsets, {"mode": trajectory_mode, "target_yaw_deg": target_yaw_deg}
    if trajectory_mode != "perturbation":
        raise ValueError(f"unsupported camera trajectory mode: {trajectory_mode}")
    trace_path = resolve_recorded_path(
        str(receipt.get("action_trace", "")), repo_root=REPO_ROOT
    )
    expected = receipt.get("action_trace_sha256")
    if not trace_path.is_file() or not isinstance(expected, str) or sha256(trace_path) != expected:
        raise ValueError(f"missing or hash-mismatched action trace: {trace_path}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    yaw_actions = [
        item
        for item in trace
        if isinstance(item, dict) and isinstance(item.get("target_yaw_deg"), (int, float))
    ]
    if len(yaw_actions) != 1:
        raise ValueError("perturbation trace must contain exactly one yaw action")
    action = yaw_actions[0]
    traced_target = float(action["target_yaw_deg"])
    if abs(traced_target - target_yaw_deg) > 1e-6:
        raise ValueError(
            f"trace target yaw {traced_target} does not match requested {target_yaw_deg}"
        )
    accepted = float(action.get("accepted_at_sec", 0.0))
    completed = float(action.get("completed_at_sec", accepted))
    timeline_end = max(
        float(item.get("completed_at_sec", completed))
        for item in trace
        if isinstance(item, dict)
    )
    if completed <= accepted or timeline_end <= 0:
        raise ValueError("invalid perturbation timing in action trace")
    times = np.arange(output_count, dtype=np.float64) / output_count * timeline_end
    fractions = np.clip((times - accepted) / (completed - accepted), 0.0, 1.0)
    return fractions * target_yaw_deg, {
        "mode": trajectory_mode,
        "target_yaw_deg": target_yaw_deg,
        "action_trace": str(trace_path),
        "action_trace_sha256": expected,
        "action_accepted_at_sec": accepted,
        "action_completed_at_sec": completed,
        "timeline_end_sec": timeline_end,
    }


def _load_camera_ground_truth(case_root: Path) -> dict[str, Any]:
    case = yaml.safe_load((case_root / "case.yaml").read_text(encoding="utf-8"))
    params = case.get("module_params", {}).get("environment_object_consistency", {})
    required = {
        "poses": "eoc_unity_poses_npy",
        "intrinsics": "eoc_unity_intrinsics_npy",
        "objects": "eoc_unity_object_transforms_json",
    }
    paths: dict[str, Path] = {}
    for name, key in required.items():
        value = params.get(key)
        if not isinstance(value, str):
            raise ValueError(f"camera remapping requires {key} in the Lite case")
        path = (case_root / value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[name] = path
    poses = np.load(paths["poses"], allow_pickle=False)
    intrinsics = np.load(paths["intrinsics"], allow_pickle=False)
    objects = json.loads(paths["objects"].read_text(encoding="utf-8"))
    if poses.shape[0] != intrinsics.shape[0]:
        raise ValueError("reference pose and intrinsics timelines differ")
    diameters = params.get("eoc_unity_object_diameters_json")
    if isinstance(diameters, str):
        diameter_path = (case_root / diameters).resolve()
        if not diameter_path.is_file():
            raise FileNotFoundError(diameter_path)
        paths["diameters"] = diameter_path
    return {
        "paths": paths,
        "poses": poses,
        "intrinsics": intrinsics,
        "objects": objects,
        "params": params,
    }


def _write_mapped_ground_truth(
    *,
    destination: Path,
    source: dict[str, Any],
    source_indices: list[int],
    mapping: dict[str, Any],
) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    poses_path = destination / "poses.npy"
    intrinsics_path = destination / "intrinsics.npy"
    objects_path = destination / "object_transforms.json"
    indices = np.asarray(source_indices, dtype=int)
    np.save(poses_path, source["poses"][indices], allow_pickle=False)
    np.save(intrinsics_path, source["intrinsics"][indices], allow_pickle=False)
    source_objects = source["objects"].get("objects")
    if not isinstance(source_objects, dict):
        raise ValueError("reference object transforms lack an objects mapping")
    mapped_objects: dict[str, Any] = {}
    for label, entry in source_objects.items():
        frames = entry.get("frames", [])
        positions = entry.get("positions_world", [])
        if len(frames) != len(positions):
            raise ValueError(f"object {label!r} has mismatched frames and positions")
        if entry.get("static") is True:
            if len(positions) != 1:
                raise ValueError(f"static object {label!r} must provide one position")
            mapped_objects[label] = {
                "frames": list(range(len(source_indices))),
                "positions_world": [positions[0] for _ in source_indices],
                "static": True,
            }
            continue
        by_source_frame = {int(frame): position for frame, position in zip(frames, positions)}
        candidate_frames: list[int] = []
        candidate_positions: list[Any] = []
        for candidate_frame, source_frame in enumerate(source_indices):
            if source_frame in by_source_frame:
                candidate_frames.append(candidate_frame)
                candidate_positions.append(by_source_frame[source_frame])
        mapped_objects[label] = {
            "frames": candidate_frames,
            "positions_world": candidate_positions,
        }
    objects_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": str(source["paths"]["objects"]),
                "timeline_mapping": "nearest_reference_pose_by_camera_forward_yaw",
                "trajectory_mapping": mapping,
                "objects": mapped_objects,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"poses": poses_path, "intrinsics": intrinsics_path, "objects": objects_path}


def standardize(
    source: Path,
    destination: Path,
    *,
    source_kind: str,
    target: dict[str, int | float],
) -> None:
    ffmpeg = media_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    source_info = probe(source)
    source_duration = float(source_info["frame_count"]) / float(source_info["fps"])
    target_duration = float(target["frame_count"]) / float(target["fps"])
    width, height = int(target["width"]), int(target["height"])
    if source_kind == "browser_capture":
        # Project Genie's live viewport includes top progress/ESC controls and
        # bottom keyboard controls. This is the same content crop used by the
        # visual native aligner, then center-fit to the Unity pixel frame.
        spatial = (
            "crop=iw*0.90:ih*0.69:iw*0.05:ih*0.09,"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    elif source_kind == "native_aligned":
        spatial = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        )
    else:
        raise ValueError(f"unsupported source kind: {source_kind}")
    temporal = (
        f"setpts={target_duration / source_duration:.12f}*PTS,"
        f"fps={float(target['fps']):.12g}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", f"{spatial},{temporal}", "-frames:v", str(target["frame_count"]),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(destination),
        ],
        check=True,
    )
    actual = probe(destination)
    expected = {
        "frame_count": int(target["frame_count"]),
        "fps": float(target["fps"]),
        "width": width,
        "height": height,
    }
    if actual != expected:
        raise ValueError(f"standardized metadata mismatch: {actual} != {expected}")


def prepare(
    run_root: Path,
    lite_case_id: str,
    output: Path,
    *,
    model: str = "project_genie",
    camera_trajectory_mode: str | None = None,
    reference_start_frame: int = 0,
    reference_target_yaw_deg: float = 360.0,
    candidate_params: dict[str, Any] | None = None,
    reference_video_override: Path | None = None,
    repeat_ids: tuple[str, ...] = REPEATS,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    output = output.resolve()
    case_root = REPO_ROOT / "lite" / "cases" / lite_case_id
    reference = (
        reference_video_override.expanduser().resolve()
        if reference_video_override is not None
        else next((path for path in (case_root / "reference").glob("*.mp4")), None)
    )
    if reference is None:
        raise FileNotFoundError(f"{lite_case_id}: no reference video")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    target = probe(reference)
    camera_gt = _load_camera_ground_truth(case_root) if camera_trajectory_mode else None
    receipt_rows: list[tuple[str, Path, dict[str, Any]]] = []
    for repeat_id in repeat_ids:
        receipt_path = run_root / repeat_id / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete" or receipt.get("repeat_id") != repeat_id:
            raise ValueError(f"{repeat_id}: receipt is not complete and matching")
        if receipt.get("generation_mode") != "new_world_generation" or receipt.get(
            "table_repeat_eligible"
        ) is not True:
            raise ValueError(f"{repeat_id}: not an independent table-eligible generation")
        receipt_rows.append((repeat_id, receipt_path, receipt))
    selected_artifacts, source_selection_policy = select_common_artifacts(
        [receipt for _, _, receipt in receipt_rows],
        [receipt_path.parent for _, receipt_path, _ in receipt_rows],
    )

    rows: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    input_hashes: set[tuple[str, str]] = set()
    for (repeat_id, receipt_path, receipt), (source, source_kind) in zip(
        receipt_rows, selected_artifacts, strict=True
    ):
        destination = output / "standardized_videos" / f"{repeat_id}.mp4"
        standardize(source, destination, source_kind=source_kind, target=target)
        input_hashes.add(
            (str(receipt.get("first_frame_sha256")), str(receipt.get("actions_sha256")))
        )
        row = {
            "model": model,
            "case_id": lite_case_id,
            "seed": repeat_id,
            "candidate_video": str(destination),
        }
        if candidate_params:
            row["params"] = dict(candidate_params)
        source_indices: list[int] | None = None
        trajectory_mapping: dict[str, Any] | None = None
        if camera_gt is not None and camera_trajectory_mode is not None:
            offsets, trajectory_mapping = _yaw_offsets_for_run(
                receipt=receipt,
                output_count=int(target["frame_count"]),
                trajectory_mode=camera_trajectory_mode,
                target_yaw_deg=reference_target_yaw_deg,
            )
            source_indices = pose_yaw_source_indices(
                camera_gt["poses"],
                start_frame=reference_start_frame,
                yaw_offsets_deg=offsets,
            )
            mapped = _write_mapped_ground_truth(
                destination=output / "ground_truth" / repeat_id,
                source=camera_gt,
                source_indices=source_indices,
                mapping={
                    **trajectory_mapping,
                    "reference_start_frame": reference_start_frame,
                },
            )
            row.setdefault("params", {}).update({
                "eoc_unity_object_transforms_json": str(mapped["objects"]),
                "eoc_unity_poses_npy": str(mapped["poses"]),
                "eoc_unity_intrinsics_npy": str(mapped["intrinsics"]),
            })
            source_params = camera_gt.get("params", {})
            for key in ("eoc_text_prompt", "eoc_fixed_anchor_labels"):
                if key in source_params:
                    row["params"][key] = source_params[key]
            if "diameters" in camera_gt["paths"]:
                row["params"]["eoc_unity_object_diameters_json"] = str(
                    camera_gt["paths"]["diameters"]
                )
            character_prompt = str(receipt.get("character_prompt", "")).strip()
            if character_prompt:
                row["params"]["eoc_character_prompt"] = character_prompt
        rows.append(row)
        prepared.append(
            {
                "repeat_id": repeat_id,
                "receipt": str(receipt_path),
                "source_kind": source_kind,
                "source": str(source),
                "source_sha256": sha256(source),
                "standardized": str(destination),
                "standardized_sha256": sha256(destination),
                "candidate_video": str(destination),
                "candidate_video_sha256": sha256(destination),
                "metadata": probe(destination),
                "source_reference_frame_indices": source_indices,
                "trajectory_mapping": trajectory_mapping,
            }
        )
    if len(input_hashes) != 1:
        raise ValueError("runs do not share one conditioning/action input pair")
    output.mkdir(parents=True, exist_ok=True)
    candidate_manifest = output / "candidate_manifest.jsonl"
    candidate_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    report = {
        "schema_version": 2,
        "status": "ready",
        "claim_scope": "paper_protocol_reconstruction_not_exact_paper_parity",
        "lite_case_id": lite_case_id,
        "run_root": str(run_root),
        "reference_video": str(reference.resolve()),
        "reference_video_sha256": sha256(reference),
        "target_metadata": target,
        "repeat_count": len(repeat_ids),
        "shared_input_hashes": list(next(iter(input_hashes))),
        "source_selection_policy": source_selection_policy,
        "camera_reference_mapping": (
            {
                "mode": camera_trajectory_mode,
                "reference_start_frame": reference_start_frame,
                "target_yaw_deg": reference_target_yaw_deg,
                "selection": "nearest_reference_pose_by_camera_forward_yaw",
            }
            if camera_trajectory_mode
            else None
        ),
        "candidate_params": candidate_params,
        "parity_limitations": [
            "The paper does not publish exact prompts, numeric action timelines, or backend adapter details.",
            "The Lite evaluator is a diagnostic proxy and does not implement every published Table 4/7 metric with paper-identical semantics.",
            "Project Genie web generations expose independent run IDs, not controllable RNG seeds.",
        ],
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "runs": prepared,
    }
    (output / "evaluation_inputs.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lite-case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="project_genie")
    parser.add_argument(
        "--camera-trajectory-mode", choices=("full_turn", "perturbation")
    )
    parser.add_argument("--reference-start-frame", type=int, default=0)
    parser.add_argument("--reference-target-yaw-deg", type=float, default=360.0)
    parser.add_argument("--repeat-count", type=int, default=5)
    args = parser.parse_args()
    report = prepare(
        args.run_root,
        args.lite_case_id,
        args.output_dir,
        model=args.model,
        camera_trajectory_mode=args.camera_trajectory_mode,
        reference_start_frame=args.reference_start_frame,
        reference_target_yaw_deg=args.reference_target_yaw_deg,
        repeat_ids=tuple(f"run{index}" for index in range(args.repeat_count)),
    )
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]), "output": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
