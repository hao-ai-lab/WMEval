"""Kinematics and Mechanics evaluation (paper appendix B.3).

Reconstructs key-object trajectories from the candidate (action-conditioned)
rollout and compares them against Unity-side references:

  1. Detect key objects on every sampled frame with GroundingDINO.
  2. Associate detections across frames into per-object tracks with a
     lightweight ByteTrack-style tracker (``bytetracker`` /
     ``cjm_byte_track`` if installed -- see ``README_kinematics_mechanics.md``
     for the exact optional dependency this repo expects, matching
     ``tools/groundingdino_bytetrack.py``).
  3. Estimate camera ego-motion between frames from the *static* anchor
     tracks (objects declared as ``kin_static_anchor_labels``) via the same
     median-translation flow-compensation strategy already used by
     :mod:`evaluation.modules.flow_utils`/``control_motion``, and subtract it
     from every tracked centroid to express trajectories in a common,
     (approximately) static reference frame.
  4. Compare the compensated candidate trajectory against a Unity ground
     truth trajectory with FDE (final displacement error) and DTW (dynamic
     time warping).
  5. Differentiate the compensated trajectory to get velocity, then combine
     with a simulator-provided mass to report momentum/kinetic-energy
     ExactMatch and ErrorRate against Unity-derived references.

Ground-truth sourcing: matching the analytical-ground-truth convention used
by :mod:`evaluation.modules.environment_object_consistency` (paper appendix
B.1) rather than re-running detectors on Unity's own reference video, this
module expects:

  - ``kin_unity_object_transforms_json``: per-frame per-object 3D world (or
    image-plane, see the README) positions -- same schema as
    ``eoc_unity_object_transforms_json``.
  - ``kin_unity_object_masses_json``: ``{"<label>": mass_kg, ...}``.

As of this writing neither artifact is bundled with any case (see
``eval_sets/wmeval_lite_v1/PENDING.md``); when they are absent this module
still reports the candidate-side reconstructed trajectories (useful for
inspection/QA) but leaves every ground-truth-dependent metric as ``None``
with ``ground_truth_status: "pending"`` rather than inventing a reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )
except ImportError:
    AutoModelForZeroShotObjectDetection = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]

try:
    from cjm_byte_track.core import BYTETracker
except ImportError:
    try:
        from bytetracker import BYTETracker  # type: ignore[no-redef]
    except ImportError:
        BYTETracker = None  # type: ignore[assignment,misc]

from ..context import EvaluationContext
from .shadow_consistency import GD_DEFAULT_MODEL, _detect_objects_groundingdino
from .environment_object_consistency import (
    _dedupe_seed_detections,
    _load_camera_intrinsics,
    _load_camera_poses,
)

EPS = 1e-6


# ---------------------------------------------------------------------------
# Ground-truth artifact loading (mirrors environment_object_consistency.py)
# ---------------------------------------------------------------------------
def _load_unity_object_transforms(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"unity object transforms artifact does not exist: {path}")
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "objects" not in payload:
        raise ValueError(f"unity object transforms artifact must be a dict with an 'objects' key: {path}")
    return payload


def _load_object_masses(path: str) -> Dict[str, float]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"unity object masses artifact does not exist: {path}")
    with p.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"unity object masses artifact must be a flat {{label: mass_kg}} dict: {path}")
    return {str(k).strip().lower(): float(v) for k, v in payload.items()}


def _unity_trajectory(gt: Dict[str, Any], label: str) -> Optional[List[Tuple[int, np.ndarray]]]:
    objects = gt.get("objects", {})
    lower_index = {str(k).strip().lower(): v for k, v in objects.items()}
    entry = lower_index.get(label.strip().lower())
    if entry is None:
        return None
    frames = entry.get("frames", [])
    positions = entry.get("positions_world", [])
    return [
        (int(f), np.asarray(pos, dtype=np.float64))
        for f, pos in zip(frames, positions)
    ]


# ---------------------------------------------------------------------------
# Candidate-side detection + tracking
# ---------------------------------------------------------------------------
def _detect_all_frames(
    video_path: str,
    text_prompt: str,
    device: str,
    gd_model_id: str,
    gd_threshold: float,
    gd_text_threshold: float,
) -> Tuple[List[Dict[str, Any]], float, Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    per_frame: List[Dict[str, Any]] = []
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, scores, labels = _detect_objects_groundingdino(
            frame, text_prompt, gd_model_id, gd_threshold, gd_text_threshold, device
        )
        # Same degenerate/duplicate-detection issue as _track_objects in
        # environment_object_consistency.py: a multi-noun prompt can produce
        # a spurious near-full-frame merged-label box and/or duplicate
        # same-label boxes, which would otherwise confuse ByteTrack's
        # nearest-centroid label assignment below.
        boxes, scores, labels = _dedupe_seed_detections(
            boxes, scores, labels, frame_width=width, frame_height=height,
        )
        per_frame.append({"frame": f_idx, "boxes": boxes, "scores": scores, "labels": labels})
        f_idx += 1
    cap.release()
    return per_frame, fps, (width, height)


def _track_with_bytetrack(
    per_frame: List[Dict[str, Any]],
    frame_size: Tuple[int, int],
    fps: float,
    track_thresh: float,
    track_buffer: int,
    match_thresh: float,
) -> Dict[int, List[Dict[str, Any]]]:
    """Associate per-frame GroundingDINO detections into tracks with
    ByteTrack. Returns {track_id: [{"frame": int, "label": str, "score":
    float, "centroid": [u, v]}, ...]}."""
    tracker = BYTETracker(
        track_thresh=track_thresh, track_buffer=track_buffer,
        match_thresh=match_thresh, frame_rate=fps,
    )
    tracks: Dict[int, List[Dict[str, Any]]] = {}
    for entry in per_frame:
        boxes, scores, labels = entry["boxes"], entry["scores"], entry["labels"]
        if len(boxes) == 0:
            continue
        dets = np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)
        raw_tracks = tracker.update(output_results=dets, img_info=frame_size, img_size=frame_size)
        for t in raw_tracks:
            tlbr = np.asarray(t.tlbr, dtype=np.float64) if hasattr(t, "tlbr") else np.asarray(t[:4])
            track_id = int(t.track_id) if hasattr(t, "track_id") else int(t[4])
            # Nearest detection box by IoU centre distance gives the label.
            centres = np.stack(
                [(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1
            )
            tcentre = np.asarray([(tlbr[0] + tlbr[2]) / 2.0, (tlbr[1] + tlbr[3]) / 2.0])
            nearest = int(np.argmin(np.linalg.norm(centres - tcentre, axis=1)))
            tracks.setdefault(track_id, []).append({
                "frame": entry["frame"],
                "label": labels[nearest],
                "score": float(scores[nearest]),
                "centroid": [float(tcentre[0]), float(tcentre[1])],
            })
    return tracks


# ---------------------------------------------------------------------------
# Camera-motion compensation using static anchors
# ---------------------------------------------------------------------------
def _compensate_camera_motion(
    tracks: Dict[int, List[Dict[str, Any]]],
    static_anchor_labels: Sequence[str],
    n_frames: int,
) -> Dict[int, List[Dict[str, Any]]]:
    anchor_lower = {l.strip().lower() for l in static_anchor_labels}
    anchor_positions_by_frame: Dict[int, List[np.ndarray]] = {}
    for pts in tracks.values():
        for p in pts:
            if p["label"].strip().lower() in anchor_lower:
                anchor_positions_by_frame.setdefault(p["frame"], []).append(
                    np.asarray(p["centroid"], dtype=np.float64)
                )

    if not anchor_positions_by_frame:
        # No anchors configured/detected: nothing to compensate, but downstream
        # code always reads "centroid_compensated", so default it to the raw
        # (uncompensated) centroid rather than omitting the key.
        return {
            tid: [{**p, "centroid_compensated": list(p["centroid"])} for p in pts]
            for tid, pts in tracks.items()
        }

    frame_ids = sorted(anchor_positions_by_frame)
    anchor_mean = {
        f: np.mean(np.stack(anchor_positions_by_frame[f]), axis=0) for f in frame_ids
    }
    reference_pos = anchor_mean[frame_ids[0]]
    camera_shift = {f: anchor_mean[f] - reference_pos for f in frame_ids}

    def _shift_for_frame(f: int) -> np.ndarray:
        if f in camera_shift:
            return camera_shift[f]
        nearest = min(camera_shift, key=lambda k: abs(k - f))
        return camera_shift[nearest]

    compensated: Dict[int, List[Dict[str, Any]]] = {}
    for tid, pts in tracks.items():
        compensated[tid] = [
            {**p, "centroid_compensated": [
                p["centroid"][0] - _shift_for_frame(p["frame"])[0],
                p["centroid"][1] - _shift_for_frame(p["frame"])[1],
            ]}
            for p in pts
        ]
    return compensated


# ---------------------------------------------------------------------------
# Trajectory-fidelity metrics
# ---------------------------------------------------------------------------
def _fde(traj_hat: np.ndarray, traj_ref: np.ndarray) -> float:
    return float(np.linalg.norm(traj_hat[-1] - traj_ref[-1]))


def _dtw(traj_hat: np.ndarray, traj_ref: np.ndarray) -> float:
    n, m = len(traj_hat), len(traj_ref)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = float(np.linalg.norm(traj_hat[i - 1] - traj_ref[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])


def _velocity_series(traj: np.ndarray, dt: float) -> np.ndarray:
    if len(traj) < 2:
        return np.empty((0, traj.shape[1] if traj.ndim > 1 else 0))
    return (traj[1:] - traj[:-1]) / dt


def _project_unity_trajectory_to_pixels(
    traj_world: List[Tuple[int, np.ndarray]],
    poses: np.ndarray,
    intrinsics: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Project a Unity 3D world trajectory into that same Unity camera's
    image plane, frame by frame, using the ground-truth camera-to-world pose
    and intrinsics already bundled with the case (``poses.npy`` /
    ``intrinsics.npy`` convention)."""
    out: Dict[int, np.ndarray] = {}
    for f, pos_w in traj_world:
        pf = min(max(f, 0), poses.shape[0] - 1)
        c2w = poses[pf]
        w2c = np.linalg.inv(c2w)
        pos_h = np.append(pos_w, 1.0)
        pos_c = w2c @ pos_h
        z = pos_c[2]
        if abs(z) < EPS:
            continue
        ifr = min(max(f, 0), intrinsics.shape[0] - 1)
        fx, fy, cx, cy = intrinsics[ifr]
        u = fx * pos_c[0] / z + cx
        v = fy * pos_c[1] / z + cy
        out[f] = np.asarray([u, v], dtype=np.float64)
    return out


def _round_sig2(value: float) -> float:
    if value == 0:
        return 0.0
    from math import floor, log10

    digits = 1 - int(floor(log10(abs(value))))
    return round(value, digits)


# ---------------------------------------------------------------------------
# Registry entry point
# ---------------------------------------------------------------------------
def analyze_kinematics_mechanics_from_context(
    context: EvaluationContext,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Entry point called from :pymod:`evaluation.registry`."""
    if torch is None:
        return {"module": "kinematics_mechanics", "error": "torch is not installed"}
    if AutoProcessor is None:
        return {"module": "kinematics_mechanics", "error": "transformers is not installed"}
    if BYTETracker is None:
        return {
            "module": "kinematics_mechanics",
            "error": (
                "ByteTrack is not installed. Install one of `cjm_byte_track` "
                "(matches tools/groundingdino_bytetrack.py) or `bytetracker`."
            ),
        }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts = context.artifacts or {}
    text_prompt = str(artifacts.get("kin_text_prompt", "ball. car. object."))
    static_anchor_labels = [
        s.strip() for s in str(artifacts.get("kin_static_anchor_labels", "")).split(",") if s.strip()
    ]
    track_thresh = float(artifacts.get("kin_track_thresh", 0.5))
    track_buffer = int(artifacts.get("kin_track_buffer", 30))
    match_thresh = float(artifacts.get("kin_match_thresh", 0.8))
    min_track_len = int(artifacts.get("kin_min_track_len", 4))
    gd_threshold = float(artifacts.get("kin_gd_threshold", 0.30))
    gd_text_threshold = float(artifacts.get("kin_gd_text_threshold", 0.25))
    gd_model_id = str(artifacts.get("kin_gd_model") or GD_DEFAULT_MODEL)

    video_path = context.candidate_video
    if not video_path:
        return {"module": "kinematics_mechanics", "error": "candidate_video is required"}

    per_frame, fps, frame_size = _detect_all_frames(
        video_path, text_prompt, device, gd_model_id, gd_threshold, gd_text_threshold
    )
    if not any(len(f["boxes"]) for f in per_frame):
        return {
            "module": "kinematics_mechanics",
            "error": "GroundingDINO found 0 objects across the whole candidate video",
        }

    raw_tracks = _track_with_bytetrack(
        per_frame, frame_size, fps, track_thresh, track_buffer, match_thresh
    )
    n_frames = len(per_frame)
    tracks = _compensate_camera_motion(raw_tracks, static_anchor_labels, n_frames)

    # Keep only tracks with enough length to be a trajectory, and take the
    # dominant label per track (mode of per-frame labels).
    kept_tracks: Dict[str, List[Dict[str, Any]]] = {}
    for tid, pts in tracks.items():
        if len(pts) < min_track_len:
            continue
        labels = [p["label"] for p in pts]
        dominant = max(set(labels), key=labels.count)
        kept_tracks.setdefault(dominant, []).extend(pts)

    trajectories_out: Dict[str, Any] = {}
    for label, pts in kept_tracks.items():
        pts_sorted = sorted(pts, key=lambda p: p["frame"])
        trajectories_out[label] = [
            {"frame": p["frame"], "centroid_uv": p["centroid"],
             "centroid_uv_compensated": p["centroid_compensated"]}
            for p in pts_sorted
        ]

    result: Dict[str, Any] = {
        "module": "kinematics_mechanics",
        "video": video_path,
        "device": device,
        "text_prompt": text_prompt,
        "static_anchor_labels": static_anchor_labels,
        "fps": fps,
        "n_frames": n_frames,
        "trajectories": trajectories_out,
        "camera_compensation_applied": bool(static_anchor_labels) and any(
            "centroid_compensated" in p for pts in tracks.values() for p in pts
        ),
    }

    gt_json_path = artifacts.get("kin_unity_object_transforms_json")
    poses_path = artifacts.get("kin_unity_poses_npy")
    intrinsics_path = artifacts.get("kin_unity_intrinsics_npy")
    masses_path = artifacts.get("kin_unity_object_masses_json")

    if not gt_json_path or not poses_path or not intrinsics_path:
        result["ground_truth_status"] = "pending"
        result["ground_truth_note"] = (
            "kin_unity_object_transforms_json / kin_unity_poses_npy / "
            "kin_unity_intrinsics_npy are not fully configured for this case; see "
            "eval_sets/wmeval_lite_v1/PENDING.md. Trajectory-fidelity (FDE/DTW) and "
            "mechanical-quantity (momentum/energy) metrics cannot be computed without "
            "a Unity-side analytical reference trajectory."
        )
        result["trajectory_fidelity"] = None
        result["mechanical_quantities"] = None
        return result

    try:
        gt = _load_unity_object_transforms(str(gt_json_path))
        poses = _load_camera_poses(str(poses_path))
        intrinsics = _load_camera_intrinsics(str(intrinsics_path))
    except Exception as exc:
        result["ground_truth_status"] = "error"
        result["ground_truth_note"] = f"failed to load Unity ground-truth artifacts: {exc}"
        result["trajectory_fidelity"] = None
        result["mechanical_quantities"] = None
        return result

    masses: Dict[str, float] = {}
    if masses_path:
        try:
            masses = _load_object_masses(str(masses_path))
        except Exception as exc:
            result["mass_artifact_error"] = str(exc)

    result["ground_truth_status"] = "available"
    fidelity: Dict[str, Any] = {}
    mechanics: Dict[str, Any] = {}
    dt = 1.0 / fps if fps > 0 else 1.0

    for label, pts in trajectories_out.items():
        traj_world = _unity_trajectory(gt, label)
        if traj_world is None or len(traj_world) < 2:
            continue

        # Project the Unity 3D world trajectory into Unity's own (ground
        # truth) camera image plane, then compare against the candidate's
        # *raw* (uncompensated) pixel trajectory -- both live in native
        # image space of a moving camera, so no extra frame alignment is
        # needed. This is a pixel-space proxy for the paper's 3D FDE/DTW:
        # without a per-frame *candidate* camera-pose artifact we cannot
        # lift the candidate trajectory into a metric 3D frame, so this
        # substitutes 2D reprojection error for full 3D trajectory error.
        ref_uv = _project_unity_trajectory_to_pixels(traj_world, poses, intrinsics)
        common_frames = sorted(set(ref_uv) & {p["frame"] for p in pts})
        if len(common_frames) < 2:
            fidelity[label] = {
                "status": "insufficient_overlap",
                "note": "no overlapping frames between candidate track and projected Unity trajectory",
            }
            continue

        cand_by_frame = {p["frame"]: np.asarray(p["centroid_uv"], dtype=np.float64) for p in pts}
        traj_hat = np.stack([cand_by_frame[f] for f in common_frames])
        traj_ref = np.stack([ref_uv[f] for f in common_frames])

        fidelity[label] = {
            "status": "reprojection_proxy_2d",
            "fde_px": round(_fde(traj_hat, traj_ref), 3),
            "dtw_px": round(_dtw(traj_hat, traj_ref), 3),
            "n_frames_compared": len(common_frames),
            "note": (
                "2D pixel-space proxy (candidate raw trajectory vs Unity trajectory "
                "reprojected through Unity's own ground-truth camera); not the "
                "paper's 3D metric FDE/DTW -- see module docstring."
            ),
        }

        mass_kg = masses.get(label.strip().lower())
        if mass_kg is None:
            mechanics[label] = {
                "status": "pending",
                "note": "no mass entry for this label in kin_unity_object_masses_json",
            }
            continue

        v_hat_series = _velocity_series(traj_hat, dt)
        if len(v_hat_series) == 0:
            continue
        speed_hat = float(np.linalg.norm(v_hat_series[-1]))
        mechanics[label] = {
            "status": "pixel_units_only",
            "note": (
                "momentum/kinetic-energy require metric-scale velocity; without a "
                "candidate camera-pose artifact, velocity here is in px/sec, not "
                "m/s, so p_hat/E_hat cannot be compared to Unity's SI-unit "
                "references. Reporting the raw terminal pixel speed for inspection."
            ),
            "estimated_terminal_speed_px_per_sec": round(speed_hat, 3),
            "mass_kg": mass_kg,
        }

    result["trajectory_fidelity"] = fidelity if fidelity else None
    result["mechanical_quantities"] = mechanics if mechanics else None
    return result
