#!/usr/bin/env python3
"""Prepare an auditable WMEval Lite manifest for Project Genie Table 7 runs.

This bridge deliberately validates the conditioning-frame provenance before it
combines a generated video with Unity analytical ground truth.  It also
resamples the Unity camera timeline to each native Project Genie clip instead
of silently clamping a 119--164 frame candidate to a 150 frame reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.provenance_paths import resolve_recorded_path


PAPER_ANCHORS = (
    "Main tree",
    "left lamp post",
    "right lamp post",
    "foreground rockery ornament",
    "background rockery ornament",
)
PAPER_ANCHOR_PROMPT = ". ".join(label.lower() for label in PAPER_ANCHORS) + "."
PAPER_ANCHOR_DETECTION_PROMPT = "tree. stone lantern. rock."
GARDEN_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _verified_path(
    path_value: Any,
    digest_value: Any,
    *,
    label: str,
    relative_to: Path | None = None,
) -> Path:
    if not isinstance(path_value, str) or not isinstance(digest_value, str):
        raise ValueError(f"{label}: missing path or SHA-256")
    path = resolve_recorded_path(
        path_value, repo_root=REPO_ROOT, relative_to=relative_to
    )
    if not path.is_file():
        raise ValueError(f"{label}: missing file {path}")
    actual = sha256(path)
    if actual != digest_value:
        raise ValueError(f"{label}: SHA-256 mismatch for {path}")
    return path


def _probe_video(path: Path) -> dict[str, int | float]:
    cap = cv2.VideoCapture(str(path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if frame_count < 2 or width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"could not read valid video metadata from {path}")
    return {"frame_count": frame_count, "width": width, "height": height, "fps": fps}


def _standardize_video(
    source: Path,
    destination: Path,
    source_info: dict[str, int | float],
    *,
    target_frame_count: int,
    target_fps: float,
    target_width: int,
    target_height: int,
    source_kind: str = "native_aligned",
) -> None:
    """Time-stretch a complete semantic action to the fixed Unity timeline.

    Frames are only dropped/duplicated (no optical-flow interpolation). The
    1280x704 Project Genie export is centered in the 1280x720 reference frame,
    avoiding geometric deformation while restoring the conditioning aspect.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to standardize evaluation videos")
    source_duration = int(source_info["frame_count"]) / float(source_info["fps"])
    target_duration = target_frame_count / target_fps
    if source_duration <= 0:
        raise ValueError(f"invalid source duration for {source}")
    speed_factor = target_duration / source_duration
    if source_kind == "browser_capture":
        spatial = (
            "crop=iw*0.90:ih*0.69:iw*0.05:ih*0.09,"
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
            f"crop={target_width}:{target_height}"
        )
    elif source_kind == "native_aligned":
        if int(source_info["width"]) > target_width or int(source_info["height"]) > target_height:
            raise ValueError(
                f"cannot losslessly pad {source_info['width']}x{source_info['height']} "
                f"into {target_width}x{target_height}"
            )
        x_offset = (target_width - int(source_info["width"])) // 2
        y_offset = (target_height - int(source_info["height"])) // 2
        spatial = f"pad={target_width}:{target_height}:{x_offset}:{y_offset}:black"
    else:
        raise ValueError(f"unsupported source kind: {source_kind}")
    video_filter = (
        f"{spatial},setpts={speed_factor:.12f}*PTS,fps={target_fps:.12g}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            video_filter,
            "-frames:v",
            str(target_frame_count),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


def _resample_reference(
    *,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    frame_count: int,
    candidate_width: int,
    candidate_height: int,
    reference_width: int,
    reference_height: int,
    source_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Unity poses must have shape (T, 4, 4), got {poses.shape}")
    if intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
        raise ValueError(f"Unity intrinsics must have shape (T, 4), got {intrinsics.shape}")
    if poses.shape[0] != intrinsics.shape[0] or poses.shape[0] < 2:
        raise ValueError("Unity poses and intrinsics must share a non-trivial timeline")
    if not np.all(np.isfinite(intrinsics)) or np.any(intrinsics[:, :2] <= 0):
        raise ValueError("Unity intrinsics contain invalid focal lengths")
    if not np.allclose(intrinsics[:, 0], intrinsics[:, 1], rtol=1e-6, atol=1e-6):
        raise ValueError(
            "Unity RenderTexture ground truth requires square-pixel intrinsics (fx == fy)"
        )
    if source_indices is None:
        indices = np.rint(np.linspace(0, poses.shape[0] - 1, frame_count)).astype(int)
    else:
        if len(source_indices) != frame_count:
            raise ValueError(
                f"reference mapping has {len(source_indices)} frames, expected {frame_count}"
            )
        indices = np.asarray(source_indices, dtype=int)
        if np.any(indices < 0) or np.any(indices >= poses.shape[0]):
            raise ValueError("reference mapping contains an out-of-range source frame")
    sampled_poses = poses[indices].copy()
    sampled_intrinsics = intrinsics[indices].copy()
    sampled_intrinsics[:, [0, 2]] *= candidate_width / reference_width
    sampled_intrinsics[:, [1, 3]] *= candidate_height / reference_height
    return sampled_poses, sampled_intrinsics, indices.tolist()


def full_turn_source_indices(
    *,
    source_count: int,
    output_count: int,
    start_frame: int,
    target_yaw_deg: float,
) -> list[int]:
    """Map a complete candidate turn onto a periodic Unity orbit.

    Unity's Garden reference contains ``source_count`` evenly spaced samples on
    [0, 360) rather than duplicating the 360-degree endpoint.  Candidate frame
    ``k`` therefore represents ``k / output_count`` of the requested turn.
    Negative Project Genie yaw walks the positive Unity orbit backwards.
    """
    if source_count < 2 or output_count < 2:
        raise ValueError("source and output timelines must contain at least two frames")
    if not (0 <= start_frame < source_count):
        raise ValueError(f"start frame {start_frame} is outside a {source_count}-frame orbit")
    progress = np.arange(output_count, dtype=np.float64) / float(output_count)
    offsets = np.rint(progress * (target_yaw_deg / 360.0) * source_count).astype(int)
    return ((start_frame + offsets) % source_count).tolist()


def perturbation_source_indices(
    *,
    source_count: int,
    output_count: int,
    start_frame: int,
    target_yaw_deg: float,
    action_trace: list[dict[str, Any]],
) -> tuple[list[int], dict[str, float]]:
    """Map a measured Project Genie perturb-and-hold action to Unity poses."""
    if source_count < 2 or output_count < 2:
        raise ValueError("source and output timelines must contain at least two frames")
    if not (0 <= start_frame < source_count):
        raise ValueError(f"start frame {start_frame} is outside a {source_count}-frame orbit")
    yaw_actions = [
        action
        for action in action_trace
        if isinstance(action, dict) and isinstance(action.get("target_yaw_deg"), (int, float))
    ]
    if len(yaw_actions) != 1:
        raise ValueError("perturbation action trace must contain exactly one target_yaw_deg action")
    action = yaw_actions[0]
    traced_target = float(action["target_yaw_deg"])
    if abs(traced_target - target_yaw_deg) > 1e-6:
        raise ValueError(
            f"action trace target yaw {traced_target} does not match requested {target_yaw_deg}"
        )
    accepted = float(action.get("accepted_at_sec", 0.0))
    completed = float(action.get("completed_at_sec", accepted))
    if completed <= accepted:
        raise ValueError("perturbation action trace has a non-positive action interval")
    timeline_end = max(
        float(item.get("completed_at_sec", completed))
        for item in action_trace
        if isinstance(item, dict)
    )
    if timeline_end <= 0:
        raise ValueError("perturbation action trace has a non-positive timeline")

    # The browser recorder stores sparse real-time observations at a fixed
    # media FPS. Standardization spreads those observations over the full
    # measured action timeline, so use the same normalized progress here.
    times = np.arange(output_count, dtype=np.float64) / float(output_count) * timeline_end
    fractions = np.clip((times - accepted) / (completed - accepted), 0.0, 1.0)
    yaws = fractions * target_yaw_deg
    offsets = np.rint(yaws / 360.0 * source_count).astype(int)
    indices = ((start_frame + offsets) % source_count).tolist()
    return indices, {
        "action_accepted_at_sec": accepted,
        "action_completed_at_sec": completed,
        "timeline_end_sec": timeline_end,
        "target_yaw_deg": target_yaw_deg,
    }


def _verified_action_trace(run: dict[str, Any], video: Path) -> tuple[list[dict[str, Any]], Path]:
    receipt_path = video.parent / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"perturbation run lacks receipt beside evaluation video: {receipt_path}")
    receipt = _read_object(receipt_path)
    action_trace_path = _verified_path(
        receipt.get("action_trace"),
        receipt.get("action_trace_sha256"),
        label=f"{run.get('repeat_id', 'run')} action trace",
        relative_to=receipt_path.parent,
    )
    value = json.loads(action_trace_path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{action_trace_path}: expected an action array")
    return value, action_trace_path


def prepare_evaluation(
    *,
    validation_path: Path,
    unity_receipt_path: Path,
    output_dir: Path,
    model: str = "project_genie",
    case_id: str = "WanderingGarden/scene_first",
    trajectory_mode: str = "full_turn",
    reference_start_frame: int = 0,
    reference_target_yaw_deg: float = 360.0,
    conditioning_frame_override: Path | None = None,
    probe_video: Callable[[Path], dict[str, int | float]] = _probe_video,
    standardize_video: Callable[..., None] = _standardize_video,
) -> dict[str, Any]:
    validation_path = validation_path.resolve()
    unity_receipt_path = unity_receipt_path.resolve()
    output_dir = output_dir.resolve()
    validation = _read_object(validation_path)
    unity = _read_object(unity_receipt_path)
    if validation.get("status") != "validated":
        raise ValueError("Project Genie validation summary is not validated")
    if unity.get("status") != "ok":
        raise ValueError("Unity capture receipt is not ok")
    if trajectory_mode not in {"full_turn", "perturbation"}:
        raise ValueError(f"unsupported trajectory mode: {trajectory_mode}")

    if conditioning_frame_override is None:
        conditioning_hash = unity.get("conditioning_frame_sha256")
        conditioning_frame = _verified_path(
            unity.get("conditioning_frame"), conditioning_hash,
            label="Unity conditioning frame", relative_to=unity_receipt_path.parent,
        )
    else:
        conditioning_frame = conditioning_frame_override.expanduser().resolve()
        if not conditioning_frame.is_file():
            raise ValueError(f"conditioning frame override is missing: {conditioning_frame}")
        conditioning_hash = sha256(conditioning_frame)
    artifacts = unity.get("evaluation_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Unity receipt has no evaluation_artifacts object")

    def artifact(name: str) -> Path:
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Unity receipt lacks evaluation artifact {name}")
        return _verified_path(
            record.get("path"), record.get("sha256"), label=name,
            relative_to=unity_receipt_path.parent,
        )

    source_transforms = artifact("object_transforms_json")
    source_poses = artifact("poses_npy")
    source_intrinsics = artifact("intrinsics_npy")
    reference_video = artifact("reference_video")
    gt = _read_object(source_transforms)
    objects = gt.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("Unity object transform artifact has no objects mapping")
    missing_anchors = [label for label in PAPER_ANCHORS if label not in objects]
    if missing_anchors:
        raise ValueError(f"Unity ground truth lacks published Garden anchors: {missing_anchors}")

    poses = np.load(source_poses, allow_pickle=False)
    intrinsics = np.load(source_intrinsics, allow_pickle=False)
    resolution = unity.get("resolution")
    if not (
        isinstance(resolution, list)
        and len(resolution) == 2
        and all(isinstance(value, (int, float)) and value > 0 for value in resolution)
    ):
        raise ValueError("Unity receipt lacks a valid [width, height] resolution")
    reference_width, reference_height = int(resolution[0]), int(resolution[1])
    target_frame_count = int(unity.get("frame_count", poses.shape[0]))
    target_fps = float(unity.get("capture_fps", 15.0))
    if target_frame_count != poses.shape[0]:
        raise ValueError("Unity receipt frame count does not match its pose timeline")

    runs = validation.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Project Genie validation summary contains no runs")
    repeat_ids = validation.get("repeat_ids")
    if not isinstance(repeat_ids, list) or len(runs) != len(repeat_ids):
        raise ValueError("Project Genie run/repeat counts disagree")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    prepared_runs: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Project Genie run entry is not an object")
        repeat_id = run.get("repeat_id")
        if not isinstance(repeat_id, str) or repeat_id not in repeat_ids:
            raise ValueError(f"invalid repeat ID in validation summary: {repeat_id!r}")
        if run.get("status") != "validated":
            raise ValueError(f"{repeat_id}: run is not validated")
        if run.get("first_frame_sha256") != conditioning_hash:
            raise ValueError(
                f"{repeat_id}: conditioning frame does not match Unity ground truth"
            )
        video = _verified_path(
            run.get("evaluation_video"),
            run.get("evaluation_video_sha256"),
            label=f"{repeat_id} evaluation video",
        )
        artifact_status = run.get("artifact_status")
        source_kind = (
            "browser_capture"
            if artifact_status == "action_aligned_capture_ready"
            else "native_aligned"
        )
        native_video_info = probe_video(video)
        standardized_video = output_dir / "standardized_videos" / f"{repeat_id}.mp4"
        standardize_video(
            video,
            standardized_video,
            native_video_info,
            target_frame_count=target_frame_count,
            target_fps=target_fps,
            target_width=reference_width,
            target_height=reference_height,
            source_kind=source_kind,
        )
        video_info = probe_video(standardized_video)
        frame_count = int(video_info["frame_count"])
        width = int(video_info["width"])
        height = int(video_info["height"])
        if (
            frame_count != target_frame_count
            or width != reference_width
            or height != reference_height
            or abs(float(video_info["fps"]) - target_fps) > 1e-6
        ):
            raise ValueError(
                f"{repeat_id}: standardized video metadata is {video_info}, expected "
                f"{target_frame_count} frames at {target_fps} fps and "
                f"{reference_width}x{reference_height}"
            )
        action_mapping: dict[str, Any]
        if trajectory_mode == "full_turn":
            source_indices = full_turn_source_indices(
                source_count=poses.shape[0],
                output_count=frame_count,
                start_frame=reference_start_frame,
                target_yaw_deg=reference_target_yaw_deg,
            )
            action_mapping = {
                "mode": "full_turn",
                "start_frame": reference_start_frame,
                "target_yaw_deg": reference_target_yaw_deg,
            }
        else:
            action_trace, action_trace_path = _verified_action_trace(run, video)
            source_indices, timing = perturbation_source_indices(
                source_count=poses.shape[0],
                output_count=frame_count,
                start_frame=reference_start_frame,
                target_yaw_deg=reference_target_yaw_deg,
                action_trace=action_trace,
            )
            action_mapping = {
                "mode": "perturbation",
                "start_frame": reference_start_frame,
                "action_trace": str(action_trace_path),
                "action_trace_sha256": sha256(action_trace_path),
                **timing,
            }
        sampled_poses, sampled_intrinsics, source_indices = _resample_reference(
            poses=poses,
            intrinsics=intrinsics,
            frame_count=frame_count,
            candidate_width=width,
            candidate_height=height,
            reference_width=reference_width,
            reference_height=reference_height,
            source_indices=source_indices,
        )

        run_dir = output_dir / "ground_truth" / repeat_id
        run_dir.mkdir(parents=True, exist_ok=True)
        poses_path = run_dir / "poses.npy"
        intrinsics_path = run_dir / "intrinsics.npy"
        transforms_path = run_dir / "object_transforms.json"
        np.save(poses_path, sampled_poses, allow_pickle=False)
        np.save(intrinsics_path, sampled_intrinsics, allow_pickle=False)
        selected_objects: dict[str, Any] = {}
        candidate_frames = list(range(frame_count))
        for label in PAPER_ANCHORS:
            source_entry = objects[label]
            positions = source_entry.get("positions_world")
            if not isinstance(positions, list) or not positions:
                raise ValueError(f"Unity anchor {label!r} has no positions_world")
            selected_objects[label] = {
                "frames": candidate_frames,
                "positions_world": [positions[min(index, len(positions) - 1)] for index in source_indices],
            }
        transforms_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": str(source_transforms),
                    "timeline_mapping": "nearest_periodic_reference_pose_for_commanded_yaw",
                    "trajectory_mapping": action_mapping,
                    "objects": selected_objects,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        params = {
            "eoc_text_prompt": PAPER_ANCHOR_DETECTION_PROMPT,
            # Lock one checkpoint for all repeats instead of allowing the
            # auto-selector to choose indoor for a single outward-facing run.
            "eoc_depth_model": GARDEN_DEPTH_MODEL,
            "eoc_fixed_anchor_labels": list(PAPER_ANCHORS),
            "eoc_character_prompt": "no_character_configured_for_this_case_xyz.",
            "eoc_max_objects": 5,
            "eoc_unity_object_transforms_json": str(transforms_path),
            "eoc_unity_poses_npy": str(poses_path),
            "eoc_unity_intrinsics_npy": str(intrinsics_path),
        }
        rows.append(
            {
                "model": model,
                "case_id": case_id,
                "seed": repeat_id,
                "candidate_video": str(standardized_video),
                "params": params,
            }
        )
        prepared_runs.append(
            {
                "repeat_id": repeat_id,
                "native_video": str(video),
                "native_video_sha256": sha256(video),
                "native_video_metadata": native_video_info,
                "source_kind": source_kind,
                "candidate_video": str(standardized_video),
                "candidate_video_sha256": sha256(standardized_video),
                "candidate_video_metadata": video_info,
                "source_reference_frame_indices": source_indices,
                "trajectory_mapping": action_mapping,
                "poses_sha256": sha256(poses_path),
                "intrinsics_sha256": sha256(intrinsics_path),
                "object_transforms_sha256": sha256(transforms_path),
            }
        )

    manifest_path = output_dir / "candidate_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 2,
        "status": "ready",
        "claim_scope": "paper_protocol_reconstruction_not_exact_paper_parity",
        "case_id": case_id,
        "model": model,
        "generation_count": len(rows),
        "conditioning_frame": str(conditioning_frame),
        "conditioning_frame_sha256": conditioning_hash,
        "paper_anchor_labels": list(PAPER_ANCHORS),
        "paper_anchor_prompt": PAPER_ANCHOR_PROMPT,
        "paper_anchor_detection_prompt": PAPER_ANCHOR_DETECTION_PROMPT,
        "reference_trajectory": {
            "mode": trajectory_mode,
            "start_frame": reference_start_frame,
            "target_yaw_deg": reference_target_yaw_deg,
            "unity_positive_yaw_direction": "increasing_reference_frame_index",
        },
        "paper_settings_source": "_arxiv__WorldModel_Eval/text/7_appendix_implementations.tex",
        "validation_summary": str(validation_path),
        "validation_summary_sha256": sha256(validation_path),
        "unity_receipt": str(unity_receipt_path),
        "unity_receipt_sha256": sha256(unity_receipt_path),
        "unity_reference_video": str(reference_video),
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_sha256": sha256(manifest_path),
        "timeline_standardization": (
            "Each validated clip is time-stretched to 150 frames at 15 fps (10 seconds). "
            "Unity poses are selected from the periodic orbit in the commanded yaw "
            "direction; perturb-and-hold runs use each run's measured action trace. Frames are "
            "dropped/duplicated without optical-flow interpolation; 1280x704 exports "
            "are centered in a 1280x720 frame without geometric scaling; audited "
            "browser fallbacks use the fixed Project Genie viewport crop. Native "
            "sources remain immutable and both source/derived hashes are recorded."
        ),
        "parity_limitations": [
            "The paper does not publish the exact Project Genie prompt strings, action rate, or web adapter.",
            "All five paper anchors retain canonical identities through Unity-projection assignment; anchors first detected after frame 0 use GroundingDINO box geometry because SAM2 cannot be seeded retroactively in the current adapter.",
            "Project Genie native exports are 1280x704 while the Unity capture is 1280x720; standardized copies use eight-pixel top/bottom padding.",
            "Project Genie web generations expose independent run IDs, not controllable RNG seeds.",
        ],
        "runs": prepared_runs,
    }
    receipt_path = output_dir / "evaluation_inputs.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--unity-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="project_genie")
    parser.add_argument("--case-id", default="WanderingGarden/scene_first")
    parser.add_argument(
        "--trajectory-mode", choices=("full_turn", "perturbation"), default="full_turn"
    )
    parser.add_argument("--reference-start-frame", type=int, default=0)
    parser.add_argument("--reference-target-yaw-deg", type=float, default=360.0)
    parser.add_argument("--conditioning-frame", type=Path)
    args = parser.parse_args()
    receipt = prepare_evaluation(
        validation_path=args.validation,
        unity_receipt_path=args.unity_receipt,
        output_dir=args.output_dir,
        model=args.model,
        case_id=args.case_id,
        trajectory_mode=args.trajectory_mode,
        reference_start_frame=args.reference_start_frame,
        reference_target_yaw_deg=args.reference_target_yaw_deg,
        conditioning_frame_override=args.conditioning_frame,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
