"""Kinematics and Mechanics evaluation (paper appendix B.3).

Reconstructs key-object trajectories from the candidate (action-conditioned)
rollout and compares them against Unity-side references:

  1. Detect key objects on every sampled frame with GroundingDINO.
  2. Associate detections across frames into per-object tracks with a
     lightweight ByteTrack-style tracker (``bytetracker`` /
     ``cjm_byte_track`` if installed -- see the "Kinematics & Mechanics" section of ``README.md``
     for the exact optional dependency this repo expects, matching
     ``tools/legacy/groundingdino_bytetrack.py``).
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

As of this writing neither artifact is bundled with any case ; when they are absent this module
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
    _auto_select_depth_model,
    _dedupe_seed_detections,
    _depth_map_for_frame,
    _load_camera_intrinsics,
    _load_camera_poses,
    _load_depth_estimator,
    _load_flat_label_json,
    _read_frame_verified,
    _unproject_pixel,
    DEPTH_DEFAULT_MODEL,
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
    """Real per-label object mass in kg. Delegates to environment_object_
    consistency.py's _load_flat_label_json, the shared loader for every
    "known physical constant per label" artifact (masses, diameters, and
    any future one) -- avoids two independently-maintained copies of the
    identical exists/parse/validate/lowercase logic."""
    return _load_flat_label_json(path, "object masses")


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
def _median_hsv(frame: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    """Median HSV color inside a detection box, used to catch a zero-shot
    detector firing on the wrong same-label object (e.g. GroundingDINO's
    "yellow ball." occasionally matching a white ball once it comes to rest
    near the yellow ball's start position -- observed on BilliardGame)."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = frame.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    return np.median(hsv, axis=0)


def _hue_distance_deg(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 180.0
    return min(d, 180.0 - d)


def _detect_all_frames(
    video_path: str,
    text_prompt: str,
    device: str,
    gd_model_id: str,
    gd_threshold: float,
    gd_text_threshold: float,
    color_filter_enabled: bool = False,
    color_hue_tolerance_deg: float = 25.0,
) -> Tuple[List[Dict[str, Any]], float, Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    seed_hue: Dict[str, float] = {}
    per_frame: List[Dict[str, Any]] = []
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, scores, labels = _detect_objects_groundingdino(
            frame, text_prompt, gd_model_id, gd_threshold, gd_text_threshold, device
        )
        if color_filter_enabled and len(boxes):
            # Filter by color BEFORE dedup, not after: dedup already
            # collapses same-label boxes to the single highest-scoring one,
            # so a same-frame false positive that happens to outscore the
            # real object would survive dedup and never reach a post-dedup
            # filter. Only labels with an already-established seed color
            # are filtered here; a label's first-ever appearance still has
            # no reference color, so it passes through untouched and gets
            # its seed established below, from whatever dedup keeps.
            keep = []
            for i, (box, label) in enumerate(zip(boxes, labels)):
                if label not in seed_hue:
                    keep.append(i)
                    continue
                hsv = _median_hsv(frame, box)
                if hsv is None or _hue_distance_deg(float(hsv[0]), seed_hue[label]) <= color_hue_tolerance_deg:
                    keep.append(i)
            boxes, scores, labels = (boxes[keep], scores[keep], [labels[i] for i in keep]) if keep else (boxes[:0], scores[:0], [])
        # Same degenerate/duplicate-detection issue as _track_objects in
        # environment_object_consistency.py: a multi-noun prompt can produce
        # a spurious near-full-frame merged-label box and/or duplicate
        # same-label boxes, which would otherwise confuse ByteTrack's
        # nearest-centroid label assignment below.
        boxes, scores, labels = _dedupe_seed_detections(
            boxes, scores, labels, frame_width=width, frame_height=height,
        )
        if color_filter_enabled:
            for box, label in zip(boxes, labels):
                if label not in seed_hue:
                    # First time we see this label (post-filter, post-dedup):
                    # trust dedup's highest-scoring pick and lock in its
                    # color signature for every later frame.
                    hsv = _median_hsv(frame, box)
                    if hsv is not None:
                        seed_hue[label] = float(hsv[0])
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
                "bbox_bottom_center": [float(tcentre[0]), float(tlbr[3])],
                "bbox_xyxy": [float(v) for v in tlbr[:4]],
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


def _dtw_mean(traj_hat: np.ndarray, traj_ref: np.ndarray) -> float:
    """Mean point distance along the minimum-total-cost DTW path.

    The paper table reports DTW values smaller than its corresponding FDEs,
    which is incompatible with an unnormalised endpoint-constrained DTW sum.
    We retain ``_dtw`` for diagnostics and expose this path-length-normalised
    value for the paper-scale comparison.
    """
    n, m = len(traj_hat), len(traj_ref)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance = float(np.linalg.norm(traj_hat[i - 1] - traj_ref[j - 1]))
            cost[i, j] = distance + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    i, j, path_length = n, m, 0
    while i > 0 or j > 0:
        path_length += 1
        candidates: list[tuple[float, int, int]] = []
        if i > 0:
            candidates.append((float(cost[i - 1, j]), i - 1, j))
        if j > 0:
            candidates.append((float(cost[i, j - 1]), i, j - 1))
        if i > 0 and j > 0:
            candidates.append((float(cost[i - 1, j - 1]), i - 1, j - 1))
        _, i, j = min(candidates, key=lambda item: item[0])
    return float(cost[n, m] / max(path_length, 1))


def _velocity_series(traj: np.ndarray, dt: float) -> np.ndarray:
    if len(traj) < 2:
        return np.empty((0, traj.shape[1] if traj.ndim > 1 else 0))
    return (traj[1:] - traj[:-1]) / dt


def _linear_velocity_fit(frame_numbers: Sequence[int], points: np.ndarray, fps: float) -> np.ndarray:
    """Least-squares linear-in-time fit velocity across several points,
    instead of a raw 2-point finite difference. A 2-point difference is
    extremely sensitive to per-frame noise (e.g. monocular-depth-map noise
    at a single pixel) since the entire velocity estimate rests on just two
    noisy samples; fitting a line's slope across a short trailing window
    averages that noise out. Falls back to a 2-point difference when fewer
    than 3 points are available (a line fit is underdetermined below that)."""
    if len(points) < 2:
        return np.zeros(points.shape[1] if points.ndim > 1 else 3)
    if len(points) == 2 or fps <= 0:
        dt_pair = max((frame_numbers[-1] - frame_numbers[0]) / fps, EPS) if fps > 0 else 1.0
        return (points[-1] - points[0]) / dt_pair
    t = np.asarray(frame_numbers, dtype=np.float64) / fps
    t_centered = t - t.mean()
    denom = float(np.sum(t_centered ** 2))
    if denom < EPS:
        return np.zeros(points.shape[1])
    p_centered = points - points.mean(axis=0, keepdims=True)
    return (t_centered @ p_centered) / denom


def _select_sustained_peak_step(
    points_by_frame: Dict[int, np.ndarray],
    fps: float,
    *,
    center_frame: Optional[int] = None,
    tolerance_frames: Optional[int] = None,
    min_neighbor_ratio: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Select a peak finite difference while rejecting one-frame spikes.

    The paper defines mechanics from ``(x_t - x_{t-1}) / dt``.  A line fit
    through a shared multi-frame window can mix moving and stopped samples
    when the generated event is shifted by only a few frames.  Candidate
    peaks may therefore move within a bounded temporal tolerance, but are
    selected solely by motion magnitude -- never by closeness to Unity.

    At least one adjacent step must retain ``min_neighbor_ratio`` of the
    peak speed.  This small persistence gate stops an isolated tracking jump
    from being reported as a physical event.
    """
    if fps <= 0:
        return None
    steps: Dict[int, Dict[str, Any]] = {}
    for frame in sorted(points_by_frame):
        previous = frame - 1
        if previous not in points_by_frame:
            continue
        velocity = (
            np.asarray(points_by_frame[frame], dtype=np.float64)
            - np.asarray(points_by_frame[previous], dtype=np.float64)
        ) * fps
        speed = float(np.linalg.norm(velocity))
        if not np.isfinite(speed):
            continue
        steps[frame] = {
            "frame": int(frame),
            "previous_frame": int(previous),
            "velocity": velocity,
            "speed_m_s": speed,
        }
    candidates = [
        (frame, step)
        for frame, step in steps.items()
        if center_frame is None
        or tolerance_frames is None
        or abs(frame - center_frame) <= tolerance_frames
    ]
    for frame, step in sorted(
        candidates, key=lambda item: (-item[1]["speed_m_s"], item[0])
    ):
        speed = step["speed_m_s"]
        if speed <= EPS:
            continue
        supporting_neighbors: List[Tuple[float, float]] = []
        for neighbor in (frame - 1, frame + 1):
            if neighbor not in steps:
                continue
            neighbor_speed = float(steps[neighbor]["speed_m_s"])
            if neighbor_speed < min_neighbor_ratio * speed or neighbor_speed <= EPS:
                continue
            cosine = float(
                np.dot(step["velocity"], steps[neighbor]["velocity"])
                / (speed * neighbor_speed)
            )
            # A detector jumping away and immediately snapping back creates
            # two large but opposite steps. Physical sustained motion retains
            # a non-negative direction cosine across at least one neighbor.
            if np.isfinite(cosine) and cosine >= 0.0:
                supporting_neighbors.append((neighbor_speed, cosine))
        if supporting_neighbors:
            support_speed, support_cosine = max(supporting_neighbors)
            return {
                **step,
                "support_speed_m_s": float(support_speed),
                "support_direction_cosine": float(support_cosine),
                "min_neighbor_ratio": float(min_neighbor_ratio),
            }
    return None


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


def _load_plane_calibration(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("plane calibration must use schema_version 1")
    anchors = payload.get("anchors")
    if payload.get("status") != "ready" or not isinstance(anchors, list) or len(anchors) < 4:
        raise ValueError("plane calibration does not contain four ready visual anchors")
    for anchor in anchors:
        if len(anchor.get("conditioning_uv", [])) != 2 or len(anchor.get("world_xz_m", [])) != 2:
            raise ValueError("plane calibration anchor is missing conditioning_uv/world_xz_m")
    return payload


def _track_plane_homographies(
    video_path: str, calibration: Dict[str, Any]
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Track Unity conditioning features and fit candidate-pixel -> Unity-XZ.

    This supplies the scale transform S described in the paper appendix.  It
    is deliberately independent of the dynamic-object detector: calibration
    features are selected outside the moving objects in Unity frame zero and
    followed with forward/backward-checked pyramidal LK optical flow.
    """
    anchors = calibration["anchors"]
    source_size = calibration.get("image_size", [])
    if len(source_size) != 2:
        raise ValueError("plane calibration image_size must be [width, height]")
    cap = cv2.VideoCapture(video_path)
    ok, previous = cap.read()
    if not ok or previous is None:
        cap.release()
        raise ValueError("candidate video has no readable first frame")
    height, width = previous.shape[:2]
    scale = np.asarray([width / float(source_size[0]), height / float(source_size[1])])
    previous_points = np.asarray(
        [np.asarray(anchor["conditioning_uv"], dtype=np.float32) * scale for anchor in anchors],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    active_world = np.asarray(
        [anchor["world_xz_m"] for anchor in anchors], dtype=np.float64
    )
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    homographies: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}

    def fit(frame_index: int) -> None:
        if len(active_world) < 4:
            return
        matrix, inliers = cv2.findHomography(
            previous_points.reshape(-1, 2),
            active_world,
            cv2.RANSAC,
            0.15,
        )
        if matrix is None or inliers is None or int(inliers.sum()) < 4:
            return
        homographies[frame_index] = matrix
        counts[frame_index] = int(inliers.sum())

    fit(0)
    frame_index = 1
    lk = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
    }
    exclusion_disks = calibration.get("dynamic_exclusion_disks", [])
    marker_world = np.asarray(calibration.get("marker_world_xz_m", []), dtype=np.float64)
    marker_search_radius = int(calibration.get("marker_search_radius_px", 24))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(active_world) >= 4:
            forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray, current_gray, previous_points, None, **lk
            )
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                current_gray, previous_gray, forward, None, **lk
            )
            fb_error = np.linalg.norm(
                backward.reshape(-1, 2) - previous_points.reshape(-1, 2), axis=1
            )
            inside = (
                (forward[:, 0, 0] >= 0)
                & (forward[:, 0, 0] < width)
                & (forward[:, 0, 1] >= 0)
                & (forward[:, 0, 1] < height)
            )
            keep = (
                forward_status.reshape(-1).astype(bool)
                & backward_status.reshape(-1).astype(bool)
                & np.isfinite(fb_error)
                & (fb_error <= 1.5)
                & inside
            )
            previous_points = forward[keep].reshape(-1, 1, 2)
            active_world = active_world[keep]
            fit(frame_index)
            # Long camera-follow clips continuously bring new road/table
            # texture into view.  Seed replacement features from the current
            # calibrated frame so the transform does not depend on a handful
            # of frame-zero points surviving for ten seconds.
            matrix = homographies.get(frame_index)
            if (
                matrix is not None
                and len(previous_points) < 80
                and calibration.get("anchor_region") != "explicit_unity_metric_markers"
            ):
                refresh_mask = (
                    np.zeros(current_gray.shape, dtype=np.uint8)
                    if marker_world.size
                    else np.full(current_gray.shape, 255, dtype=np.uint8)
                )
                if marker_world.size:
                    world_to_image = np.linalg.inv(matrix)
                    marker_h = np.concatenate(
                        [marker_world, np.ones((len(marker_world), 1), dtype=np.float64)], axis=1
                    )
                    projected = (world_to_image @ marker_h.T).T
                    projected = projected[:, :2] / projected[:, 2:3]
                    for point in projected:
                        if np.all(np.isfinite(point)):
                            cv2.circle(
                                refresh_mask,
                                tuple(np.rint(point).astype(int)),
                                int(round(marker_search_radius * float(scale.mean()))),
                                255,
                                -1,
                            )
                margin = max(8, min(width, height) // 60)
                refresh_mask[:margin, :] = 0
                refresh_mask[-margin:, :] = 0
                refresh_mask[:, :margin] = 0
                refresh_mask[:, -margin:] = 0
                for disk in exclusion_disks:
                    center = np.asarray(disk.get("center_uv", [0, 0]), dtype=np.float64) * scale
                    radius = int(round(float(disk.get("radius_px", 0)) * float(scale.mean())))
                    cv2.circle(
                        refresh_mask, tuple(np.rint(center).astype(int)), radius, 0, -1
                    )
                for point in previous_points.reshape(-1, 2):
                    cv2.circle(
                        refresh_mask, tuple(np.rint(point).astype(int)), 12, 0, -1
                    )
                fresh = cv2.goodFeaturesToTrack(
                    current_gray,
                    maxCorners=80 - len(previous_points),
                    qualityLevel=0.015,
                    minDistance=12,
                    mask=refresh_mask,
                    blockSize=7,
                    useHarrisDetector=False,
                )
                if fresh is not None:
                    fresh_uv = fresh.reshape(-1, 2).astype(np.float64)
                    homogeneous = np.concatenate(
                        [fresh_uv, np.ones((len(fresh_uv), 1), dtype=np.float64)], axis=1
                    )
                    mapped = (matrix @ homogeneous.T).T
                    mapped = mapped[:, :2] / mapped[:, 2:3]
                    finite = np.all(np.isfinite(mapped), axis=1)
                    if np.any(finite):
                        previous_points = np.concatenate(
                            [previous_points, fresh_uv[finite].astype(np.float32).reshape(-1, 1, 2)],
                            axis=0,
                        )
                        active_world = np.concatenate(
                            [active_world, mapped[finite]], axis=0
                        )
        previous_gray = current_gray
        frame_index += 1
    cap.release()
    return homographies, {
        "status": "ready" if homographies else "no_valid_homography",
        "initial_anchor_count": len(anchors),
        "calibrated_frame_count": len(homographies),
        "minimum_inliers": min(counts.values()) if counts else 0,
        "maximum_inliers": max(counts.values()) if counts else 0,
        "unit": "meter",
        "output_axes": ["x", "z"],
    }


def _map_pixels_to_plane(
    points_by_frame: Dict[int, np.ndarray], homographies: Dict[int, np.ndarray]
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for frame, point in points_by_frame.items():
        matrix = homographies.get(frame)
        if matrix is None:
            continue
        homogeneous = matrix @ np.asarray([point[0], point[1], 1.0])
        if abs(homogeneous[2]) <= EPS:
            continue
        out[frame] = homogeneous[:2] / homogeneous[2]
    return out


def _map_pixels_to_unity_plane_by_pose(
    points_by_frame: Dict[int, np.ndarray],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    plane_y: float,
) -> Dict[int, np.ndarray]:
    """Intersect candidate image rays with a calibrated Unity horizontal plane.

    This is the deterministic metric fallback for matched-action captures.  It
    uses the Unity reference camera at the same timestamp; visual-anchor flow
    remains a reported QA signal for camera divergence rather than silently
    allowing a drifting image homography to redefine world scale.
    """
    out: Dict[int, np.ndarray] = {}
    for frame, point in points_by_frame.items():
        pose = poses[min(max(frame, 0), len(poses) - 1)]
        fx, fy, cx, cy = intrinsics[min(max(frame, 0), len(intrinsics) - 1)]
        ray_camera = np.asarray(
            [(point[0] - cx) / fx, (point[1] - cy) / fy, 1.0], dtype=np.float64
        )
        ray_world = pose[:3, :3] @ ray_camera
        origin = pose[:3, 3]
        if abs(ray_world[1]) <= EPS:
            continue
        scale = (plane_y - origin[1]) / ray_world[1]
        if scale <= 0:
            continue
        world = origin + scale * ray_world
        out[frame] = world[[0, 2]]
    return out


def _round_sig2(value: float) -> float:
    if value == 0:
        return 0.0
    from math import floor, log10

    digits = 1 - int(floor(log10(abs(value))))
    return round(value, digits)


def _estimate_metric_points(
    cap: Any,
    frames_uv: Dict[int, np.ndarray],
    intrinsics: np.ndarray,
    depth_pipe: Any,
) -> Dict[int, np.ndarray]:
    """Lift a candidate's tracked pixel centroids to metric 3D points via
    monocular depth estimation (same model/approach as
    environment_object_consistency.py's candidate-side depth), so that
    velocity -- and therefore momentum/kinetic-energy (paper Eq 11) -- can be
    computed in real units instead of pixels/sec. Only requested frames are
    depth-estimated (not every frame in the video) to keep this affordable.
    ``depth_pipe`` is loaded once by the caller and shared across every
    tracked label in a case, not reloaded per call. ``cap`` is likewise an
    already-open ``cv2.VideoCapture`` shared across every label and both
    stages (terminal + peak_motion), not reopened per call -- opening a
    fresh capture per call was real, measurable redundant work once two
    stages started calling this per label.

    Inherits the same depth-estimator behavior (and, for a relative-depth
    model, the same domain-gap risk) already documented for
    environment_object_consistency's depth_absrel_pct -- see
    DEPTH_DEFAULT_MODEL's docstring for the metric-vs-relative distinction.
    """
    out: Dict[int, np.ndarray] = {}
    for f_idx in sorted(frames_uv):
        # Seek directly to each wanted frame instead of decoding every
        # frame of the video sequentially -- callers only ever ask for a
        # handful of frames (a trailing/peak window), and with two stages
        # (terminal + peak_motion) now calling this per label, decoding the
        # full video on every call would double an already-expensive cost
        # for no benefit. _read_frame_verified guards the correctness risk
        # this introduces (unverified seeking is not always frame-exact).
        frame = _read_frame_verified(cap, f_idx)
        if frame is None:
            continue
        u, v = frames_uv[f_idx]
        depth_map = _depth_map_for_frame(depth_pipe, frame)
        ifr = min(max(f_idx, 0), intrinsics.shape[0] - 1)
        fx, fy, cx, cy = intrinsics[ifr]
        out[f_idx] = _unproject_pixel(u, v, depth_map, fx, fy, cx, cy)
    return out


# The ONE physical constant this module's near-zero handling derives from:
# below this ground-truth speed, an object isn't meaningfully in motion.
# Every other near-zero threshold in this module (the momentum/energy
# baseline check below, and the reference-scale gate in _mechanics_stage)
# is computed FROM this single speed constant and the object's own real
# mass, rather than being independently-chosen absolute values in
# momentum/energy units -- two unrelated hardcoded numbers can silently
# drift out of sync (tuned for one quantity, forgotten for the other);
# one physical speed threshold, converted through p=mv / E=0.5*m*v^2 per
# quantity and per object's real mass, cannot.
NEAR_ZERO_SPEED_M_S = 0.01


def _mechanical_quantities_exact_match(
    q_hat: Dict[str, float], q_gt: Dict[str, float],
    denominator: Optional[Dict[str, float]] = None,
    near_zero_threshold: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Paper Eq 12/13: ExactMatch% (rounded to 2 significant digits) and mean
    relative ErrorRate%, both over the object set ``O`` (here, whichever
    tracked labels have both an estimate and a ground-truth value).

    By default (``denominator=None``) this is exactly the paper's Eq 13
    formula (divide by |q_gt|, for paper fidelity). Pass ``denominator`` to
    instead divide by a per-label reference scale (e.g. the object's own
    peak momentum/energy across its full ground-truth trajectory) -- see
    ``analyze_kinematics_mechanics_from_context`` for why: near a
    near-motionless ground-truth instant, dividing by |q_gt| explodes
    regardless of how small the absolute error is, which is a real,
    fixable weakness of evaluating only at one (possibly resting) instant,
    not something inherent to the mechanics being measured.

    A third return value carries diagnostics -- per-label absolute error
    and a flag for near-zero ground-truth baselines -- useful even when a
    reference-scale denominator is used, since |q_gt| near zero still
    means "this instant isn't a meaningful test of momentum/energy
    accuracy" regardless of what the error is measured relative to."""
    labels = sorted(set(q_hat) & set(q_gt))
    if not labels:
        return float("nan"), float("nan"), {}
    matches = 0
    rel_errors = []
    abs_errors: Dict[str, float] = {}
    near_zero_labels: List[str] = []
    for label in labels:
        qh, qg = q_hat[label], q_gt[label]
        if _round_sig2(qh) == _round_sig2(qg):
            matches += 1
        denom = denominator[label] if denominator and label in denominator else abs(qg)
        rel_errors.append(abs(qh - qg) / (denom + EPS))
        abs_errors[label] = round(abs(qh - qg), 6)
        threshold = (
            near_zero_threshold[label]
            if near_zero_threshold and label in near_zero_threshold
            else EPS
        )
        if abs(qg) < threshold:
            near_zero_labels.append(label)
    exact_match_pct = 100.0 * matches / len(labels)
    error_rate_pct = 100.0 * float(np.mean(rel_errors))
    diagnostics: Dict[str, Any] = {"absolute_errors": abs_errors}
    if near_zero_labels:
        diagnostics["near_zero_ground_truth_baseline_labels"] = near_zero_labels
        diagnostics["note"] = (
            "error_rate_pct for these labels divides by a near-zero ground-truth "
            "value (Eq 13's own formula, unmodified here for paper fidelity), "
            "inflating the relative error astronomically regardless of how small "
            "the absolute error is -- read absolute_errors for these labels "
            "instead of error_rate_pct."
        )
    return exact_match_pct, error_rate_pct, diagnostics


# Below this ground-truth peak speed, an object doesn't move meaningfully
# anywhere in its own rollout, so there is no physically real reference
# scale to normalize against either -- error_rate_pct_normalized would
# just be a smaller-but-still-fake number, not a genuinely meaningful one.
# Report it as not applicable rather than compute a misleadingly precise
# figure for an object that was never really in motion. See
# NEAR_ZERO_SPEED_M_S above -- this reuses the same single physical
# constant rather than an independently-chosen momentum/energy value.


def _mechanics_values_from_speeds(
    label: str,
    mass_kg: float,
    speed_hat: float,
    speed_gt: float,
    ref_momentum: float,
    ref_energy: float,
    peak_speed: float,
) -> Dict[str, Any]:
    """Evaluate Eq. 11--13 once both trajectories have metric speeds."""
    p_hat = mass_kg * speed_hat
    p_gt = mass_kg * speed_gt
    e_hat = 0.5 * mass_kg * speed_hat ** 2
    e_gt = 0.5 * mass_kg * speed_gt ** 2

    near_zero_momentum = mass_kg * NEAR_ZERO_SPEED_M_S
    near_zero_energy = 0.5 * mass_kg * NEAR_ZERO_SPEED_M_S ** 2
    mom_match, mom_err, mom_diag = _mechanical_quantities_exact_match(
        {label: p_hat}, {label: p_gt}, near_zero_threshold={label: near_zero_momentum}
    )
    e_match, e_err, e_diag = _mechanical_quantities_exact_match(
        {label: e_hat}, {label: e_gt}, near_zero_threshold={label: near_zero_energy}
    )
    object_never_moves = peak_speed < NEAR_ZERO_SPEED_M_S
    if object_never_moves:
        mom_err_norm: Optional[float] = None
        e_err_norm: Optional[float] = None
    else:
        _, mom_err_norm, _ = _mechanical_quantities_exact_match(
            {label: p_hat}, {label: p_gt}, denominator={label: ref_momentum}
        )
        _, e_err_norm, _ = _mechanical_quantities_exact_match(
            {label: e_hat}, {label: e_gt}, denominator={label: ref_energy}
        )
    normalized_note = (
        {"error_rate_pct_normalized_note": (
            "not applicable: this object's peak ground-truth speed "
            f"({peak_speed:.4f} m/s) never exceeds {NEAR_ZERO_SPEED_M_S} "
            "m/s anywhere in its own rollout, so there's no real reference "
            "scale to normalize against either -- see absolute_errors instead."
        )} if object_never_moves else {}
    )
    return {
        "momentum_kg_m_s": {
            "estimated": round(p_hat, 4), "ground_truth": round(p_gt, 4),
            "exact_match_pct": mom_match,
            "error_rate_pct": round(mom_err, 2),
            "error_rate_pct_normalized": round(mom_err_norm, 2) if mom_err_norm is not None else None,
            **mom_diag, **normalized_note,
        },
        "kinetic_energy_j": {
            "estimated": round(e_hat, 4), "ground_truth": round(e_gt, 4),
            "exact_match_pct": e_match,
            "error_rate_pct": round(e_err, 2),
            "error_rate_pct_normalized": round(e_err_norm, 2) if e_err_norm is not None else None,
            **e_diag, **normalized_note,
        },
    }


def _mechanics_values_from_points(
    label: str,
    mass_kg: float,
    usable: Sequence[int],
    hat_pts: np.ndarray,
    gt_pts: np.ndarray,
    fps: float,
    ref_momentum: float,
    ref_energy: float,
    peak_speed: float,
) -> Dict[str, Any]:
    """Evaluate Eq. 11--13 from aligned metric-space position samples.

    The source of ``hat_pts`` is deliberately independent of this helper:
    planar scenes use calibrated camera-ray/plane intersections, while
    non-planar scenes may still use monocular depth. This keeps the paper's
    mechanical calculation identical once positions are expressed in metres.
    """
    v_hat = _linear_velocity_fit(usable, hat_pts, fps)
    v_gt = _linear_velocity_fit(usable, gt_pts, fps)
    values = _mechanics_values_from_speeds(
        label,
        mass_kg,
        float(np.linalg.norm(v_hat)),
        float(np.linalg.norm(v_gt)),
        ref_momentum,
        ref_energy,
        peak_speed,
    )
    values["frames_used"] = [int(f) for f in usable]
    values["estimator"] = "multi_frame_linear_fit"
    return values


def _aligned_peak_event_stage(
    label: str,
    mass_kg: float,
    candidate_points: Dict[int, np.ndarray],
    ground_truth_points: Dict[int, np.ndarray],
    fps: float,
    *,
    tolerance_frames: int,
    min_neighbor_ratio: float,
) -> Optional[Dict[str, Any]]:
    """Compare independently detected, temporally corresponding peak events.

    This uses the paper's one-step finite difference and permits only a small,
    predeclared timing offset.  It never searches for the candidate speed that
    best matches Unity; it selects the strongest sustained candidate motion in
    the allowed event window.
    """
    ground_truth_peak = _select_sustained_peak_step(
        ground_truth_points, fps, min_neighbor_ratio=min_neighbor_ratio
    )
    if ground_truth_peak is None:
        return None
    candidate_peak = _select_sustained_peak_step(
        candidate_points,
        fps,
        center_frame=int(ground_truth_peak["frame"]),
        tolerance_frames=max(0, int(tolerance_frames)),
        min_neighbor_ratio=min_neighbor_ratio,
    )
    if candidate_peak is None:
        return None
    peak_speed = float(ground_truth_peak["speed_m_s"])
    values = _mechanics_values_from_speeds(
        label,
        mass_kg,
        float(candidate_peak["speed_m_s"]),
        peak_speed,
        mass_kg * peak_speed,
        0.5 * mass_kg * peak_speed ** 2,
        peak_speed,
    )
    values.update(
        {
            "estimator": "aligned_peak_finite_difference",
            "candidate_frames_used": [
                int(candidate_peak["previous_frame"]),
                int(candidate_peak["frame"]),
            ],
            "ground_truth_frames_used": [
                int(ground_truth_peak["previous_frame"]),
                int(ground_truth_peak["frame"]),
            ],
            "candidate_peak_frame": int(candidate_peak["frame"]),
            "ground_truth_peak_frame": int(ground_truth_peak["frame"]),
            "temporal_offset_frames": int(
                candidate_peak["frame"] - ground_truth_peak["frame"]
            ),
            "temporal_offset_sec": round(
                float(candidate_peak["frame"] - ground_truth_peak["frame"]) / fps,
                6,
            ),
            "candidate_speed_m_s": round(float(candidate_peak["speed_m_s"]), 6),
            "ground_truth_speed_m_s": round(peak_speed, 6),
            "event_alignment_tolerance_frames": max(0, int(tolerance_frames)),
            "persistence_min_neighbor_ratio": float(min_neighbor_ratio),
        }
    )
    return values


def _mechanics_stage(
    label: str,
    mass_kg: float,
    frames_window: List[int],
    cand_by_frame: Dict[int, np.ndarray],
    world_by_frame: Dict[int, np.ndarray],
    cap: Any,
    intrinsics: np.ndarray,
    depth_pipe: Any,
    fps: float,
    ref_momentum: float,
    ref_energy: float,
    peak_speed: float,
) -> Optional[Dict[str, Any]]:
    """Momentum/kinetic-energy (paper Eq 11) for one evaluation window,
    reported two ways: `error_rate_pct` (Eq 13, unmodified, divides by
    |q_gt|) and `error_rate_pct_normalized` (divides by this object's own
    peak momentum/energy across its FULL ground-truth trajectory instead --
    see analyze_kinematics_mechanics_from_context for why). ``cap`` is an
    already-open VideoCapture shared across every label/stage by the
    caller, not opened fresh here."""
    frames_available = [f for f in frames_window if f in cand_by_frame]
    metric_pts = _estimate_metric_points(
        cap, {f: cand_by_frame[f] for f in frames_available}, intrinsics, depth_pipe,
    )
    usable = [f for f in frames_window if f in metric_pts and f in world_by_frame]
    if len(usable) < 2:
        return None
    hat_pts = np.stack([metric_pts[f] for f in usable])
    gt_pts = np.stack([world_by_frame[f] for f in usable])
    return _mechanics_values_from_points(
        label, mass_kg, usable, hat_pts, gt_pts, fps,
        ref_momentum, ref_energy, peak_speed,
    )


def _summarize_mechanical_quantities(
    mechanics: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Collapse label/quantity mechanics into the two Table-4 columns.

    Each valid ``label × quantity`` pair is one member of the paper's set
    :math:`O`. The bounded, event-aligned finite-difference stage is selected
    when available, followed by the legacy peak window and then terminal as
    deterministic fallbacks. The reported Error Rate is the literal Eq. 13
    relative error. The peak-reference-normalized variant remains a separate
    diagnostic and is never substituted into the paper column.
    """
    exact_values: List[float] = []
    error_values: List[float] = []
    normalized_values: List[float] = []
    selections: Dict[str, str] = {}
    quantity_rows: List[Dict[str, Any]] = []
    for label, entry in sorted(mechanics.items()):
        if not isinstance(entry, dict):
            continue
        stages = entry.get("stages")
        if not isinstance(stages, dict):
            continue
        stage_name = next(
            (
                name
                for name in ("aligned_peak_event", "peak_motion", "terminal")
                if name in stages
            ),
            "terminal",
        )
        stage = stages.get(stage_name)
        if not isinstance(stage, dict):
            continue
        selections[label] = stage_name
        for quantity in ("momentum_kg_m_s", "kinetic_energy_j"):
            values = stage.get(quantity)
            if not isinstance(values, dict):
                continue
            exact = values.get("exact_match_pct")
            error = values.get("error_rate_pct")
            normalized = values.get("error_rate_pct_normalized")
            if isinstance(exact, (int, float)) and np.isfinite(float(exact)):
                exact_values.append(float(exact))
            if isinstance(error, (int, float)) and np.isfinite(float(error)):
                error_values.append(float(error))
            if isinstance(normalized, (int, float)) and np.isfinite(float(normalized)):
                normalized_values.append(float(normalized))
            quantity_rows.append(
                {
                    "label": label,
                    "stage": stage_name,
                    "quantity": quantity,
                    "estimated": values.get("estimated"),
                    "ground_truth": values.get("ground_truth"),
                    "exact_match_pct": exact,
                    "error_rate_pct": error,
                    "error_rate_pct_normalized": normalized,
                }
            )
    if not exact_values or not error_values:
        return None
    return {
        "status": "computed_from_unity_mass_and_metric_trajectories",
        "stage_policy": "aligned_peak_event_then_peak_motion_then_terminal",
        "quantity_count": len(quantity_rows),
        "exact_match_pct": round(float(np.mean(exact_values)), 6),
        "error_rate_pct": round(float(np.mean(error_values)), 6),
        "error_rate_pct_normalized": (
            round(float(np.mean(normalized_values)), 6)
            if normalized_values else None
        ),
        "selected_stage_by_label": selections,
        "quantities": quantity_rows,
    }


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
                "(matches tools/legacy/groundingdino_bytetrack.py) or `bytetracker`."
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
    color_filter_enabled = bool(artifacts.get("kin_color_filter", False))
    color_hue_tolerance_deg = float(artifacts.get("kin_color_hue_tolerance_deg", 25.0))
    track_point = str(artifacts.get("kin_track_point", "centroid"))
    if track_point not in {"centroid", "bbox_bottom_center"}:
        return {
            "module": "kinematics_mechanics",
            "error": "kin_track_point must be centroid or bbox_bottom_center",
        }
    mechanics_estimator = str(
        artifacts.get("kin_mechanics_estimator", "multi_frame_linear_fit")
    )
    if mechanics_estimator not in {
        "multi_frame_linear_fit",
        "aligned_peak_finite_difference",
    }:
        return {
            "module": "kinematics_mechanics",
            "error": (
                "kin_mechanics_estimator must be multi_frame_linear_fit or "
                "aligned_peak_finite_difference"
            ),
        }
    event_alignment_tolerance_frames = max(
        0, int(artifacts.get("kin_event_alignment_tolerance_frames", 15))
    )
    event_min_neighbor_ratio = float(
        artifacts.get("kin_event_min_neighbor_ratio", 0.5)
    )
    if not 0.0 <= event_min_neighbor_ratio <= 1.0:
        return {
            "module": "kinematics_mechanics",
            "error": "kin_event_min_neighbor_ratio must be between 0 and 1",
        }

    video_path = context.candidate_video
    if not video_path:
        return {"module": "kinematics_mechanics", "error": "candidate_video is required"}

    per_frame, fps, frame_size = _detect_all_frames(
        video_path, text_prompt, device, gd_model_id, gd_threshold, gd_text_threshold,
        color_filter_enabled=color_filter_enabled,
        color_hue_tolerance_deg=color_hue_tolerance_deg,
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
             "centroid_uv_compensated": p["centroid_compensated"],
             "bbox_bottom_center_uv": p.get("bbox_bottom_center", p["centroid"]),
             "bbox_xyxy": p.get("bbox_xyxy")}
            for p in pts_sorted
        ]

    result: Dict[str, Any] = {
        "module": "kinematics_mechanics",
        "video": video_path,
        "device": device,
        "text_prompt": text_prompt,
        "track_point": track_point,
        "mechanics_estimator": mechanics_estimator,
        "event_alignment_tolerance_frames": event_alignment_tolerance_frames,
        "event_min_neighbor_ratio": event_min_neighbor_ratio,
        "static_anchor_labels": static_anchor_labels,
        "fps": fps,
        "n_frames": n_frames,
        "trajectories": trajectories_out,
        "camera_compensation_applied": bool(static_anchor_labels) and any(
            "centroid_compensated" in p for pts in tracks.values() for p in pts
        ),
    }
    required_labels = [
        label.strip()
        for label in text_prompt.split(".")
        if label.strip()
    ]

    gt_json_path = artifacts.get("kin_unity_object_transforms_json")
    poses_path = artifacts.get("kin_unity_poses_npy")
    intrinsics_path = artifacts.get("kin_unity_intrinsics_npy")
    masses_path = artifacts.get("kin_unity_object_masses_json")
    plane_calibration_path = artifacts.get("kin_plane_calibration_json")

    if not gt_json_path or not poses_path or not intrinsics_path:
        result["ground_truth_status"] = "pending"
        result["ground_truth_note"] = (
            "kin_unity_object_transforms_json / kin_unity_poses_npy / "
            "kin_unity_intrinsics_npy are not fully configured for this case. "
            "Trajectory-fidelity (FDE/DTW) and "
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
    plane_homographies: Dict[int, np.ndarray] = {}
    metric_plane_y: Optional[float] = None
    loaded_plane_calibration: Optional[Dict[str, Any]] = None
    if plane_calibration_path:
        try:
            plane_calibration = _load_plane_calibration(str(plane_calibration_path))
            metric_plane_y = float(plane_calibration["plane"]["value_m"])
            loaded_plane_calibration = plane_calibration
            plane_homographies, plane_diagnostics = _track_plane_homographies(
                video_path, plane_calibration
            )
            result["plane_calibration"] = plane_diagnostics
            result["plane_calibration"]["metric_mapping"] = (
                "timestamp_matched_unity_camera_ray_plane_with_visual_anchor_qa"
            )
            result["plane_calibration"]["visual_anchor_role"] = (
                "camera_divergence_and_calibration_quality_assurance"
            )
        except Exception as exc:
            result["plane_calibration"] = {
                "status": "error",
                "error": str(exc),
            }
    else:
        result["plane_calibration"] = {
            "status": "not_configured",
            "note": "kin_plane_calibration_json is required for meter-scale DTW/FDE",
        }
    dt = 1.0 / fps if fps > 0 else 1.0
    depth_lift_enabled = bool(artifacts.get("kin_depth_lift", False))
    # max(2, ...) guards two misconfigurations: 0 (or negative), where
    # Python's list[-0:] is list[0:] -- the WHOLE list, not an empty
    # trailing window, silently ballooning "terminal" to the full
    # trajectory instead of failing loudly -- and 1, which is equally
    # degenerate in a different way: _mechanics_stage always needs >=2
    # points to fit a velocity, so a 1-frame window can never succeed and
    # would silently fall back to pixel_units_only for every label.
    depth_lift_window = max(2, int(artifacts.get("kin_depth_lift_window", 6)))
    depth_pipe = None
    if depth_lift_enabled:
        try:
            kin_depth_model_override = artifacts.get("kin_depth_model")
            if kin_depth_model_override:
                depth_model_id = str(kin_depth_model_override)
                result["depth_model_selection"] = "explicit_override"
            else:
                # Same auto-selection as environment_object_consistency.py's
                # eoc_depth_model (see _auto_select_depth_model's docstring):
                # test candidate checkpoints against a real, known Unity
                # depth instead of a human guessing indoor vs. outdoor per
                # scene. Reshape this module's own {label: [{"frame",
                # "centroid_uv"}]} trajectories into the {"frames": [{
                # "frame", "objects": [...]}]} shape _auto_select_depth_model
                # expects, rather than a second implementation.
                frame_to_objects: Dict[int, List[Dict[str, Any]]] = {}
                for lbl, pts in trajectories_out.items():
                    for p in pts:
                        frame_to_objects.setdefault(p["frame"], []).append(
                            {"label": lbl, "centroid_uv": p["centroid_uv"]}
                        )
                pseudo_track = {
                    "frames": [
                        {"frame": f, "objects": objs}
                        for f, objs in sorted(frame_to_objects.items())
                    ]
                }
                depth_model_id = _auto_select_depth_model(
                    video_path, pseudo_track, gt, poses, intrinsics, device
                )
                result["depth_model_selection"] = "auto"
            result["depth_model_used"] = depth_model_id
            depth_pipe = _load_depth_estimator(depth_model_id, device)
        except Exception as exc:
            result["depth_lift_error"] = (
                f"kin_depth_lift was requested but the depth model failed to load: {exc}. "
                "Falling back to pixel_units_only for every label's mechanical_quantities."
            )
            depth_lift_enabled = False

    # One shared, already-open capture for every label and both stages
    # (terminal + peak_motion) below, instead of _estimate_metric_points
    # opening a fresh cv2.VideoCapture per call -- for N labels that was
    # up to 2N redundant container opens of the same file. Released just
    # before this function returns, below; nothing in the loop body
    # returns early, so no try/finally is needed for that release to run.
    metric_points_cap = cv2.VideoCapture(video_path) if depth_lift_enabled else None

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
        cand_by_frame = {
            p["frame"]: np.asarray(
                p["bbox_bottom_center_uv"] if track_point == "bbox_bottom_center" else p["centroid_uv"],
                dtype=np.float64,
            )
            for p in pts
        }
        common_frames = sorted(set(ref_uv) & {p["frame"] for p in pts})
        if len(common_frames) < 2:
            fidelity[label] = {
                "status": "insufficient_overlap",
                "note": "no overlapping frames between candidate track and projected Unity trajectory",
            }
            continue

        traj_hat = np.stack([cand_by_frame[f] for f in common_frames])
        traj_ref = np.stack([ref_uv[f] for f in common_frames])
        fidelity_entry: Dict[str, Any] = {
            "status": "reprojection_proxy_2d",
            "fde_px": round(_fde(traj_hat, traj_ref), 3),
            "dtw_px": round(_dtw(traj_hat, traj_ref), 3),
            "n_frames_compared": len(common_frames),
            "note": (
                "2D pixel-space diagnostic (candidate trajectory vs Unity trajectory "
                "reprojected through Unity's camera)."
            ),
        }
        candidate_xz_pose = (
            _map_pixels_to_unity_plane_by_pose(
                cand_by_frame, poses, intrinsics, metric_plane_y
            )
            if metric_plane_y is not None
            else {}
        )
        candidate_xz_anchor = _map_pixels_to_plane(
            cand_by_frame, plane_homographies
        )
        # The Unity camera pose/intrinsics and horizontal plane define the
        # metric reference frame and scale deterministically.  Optical-flow
        # homographies are deliberately QA-only: long generated rollouts can
        # replace tracked features recursively, so treating that drifting
        # estimate as ground truth can create singular, kilometre-scale
        # trajectories even when the underlying pixel track is stable.
        candidate_xz = candidate_xz_pose
        metric_mapping_source = "timestamp_matched_unity_camera_ray_plane"
        reference_xz = {frame: position[[0, 2]] for frame, position in traj_world}
        metric_frames = sorted(set(candidate_xz) & set(reference_xz))
        if len(metric_frames) >= 2:
            metric_hat = np.stack([candidate_xz[frame] for frame in metric_frames])
            metric_ref = np.stack([reference_xz[frame] for frame in metric_frames])
            fidelity_entry.update(
                {
                    "status": "metric_planar_unity_camera_calibrated",
                    "fde_m": round(_fde(metric_hat, metric_ref), 6),
                    "dtw_m_mean": round(_dtw_mean(metric_hat, metric_ref), 6),
                    "dtw_m_sum": round(_dtw(metric_hat, metric_ref), 6),
                    "metric_frames_compared": len(metric_frames),
                    "metric_axes": ["unity_x", "unity_z"],
                    "metric_unit": "meter",
                    "metric_mapping_source": metric_mapping_source,
                    "visual_anchor_qa_mapped_frame_count": len(candidate_xz_anchor),
                    "pose_mapped_frame_count": len(candidate_xz_pose),
                    "note": (
                        "Candidate pixels are mapped into the timestamp-matched Unity "
                        "camera's horizontal X/Z plane. Tracked static conditioning "
                        "anchors are retained as camera-divergence QA and never redefine "
                        "metric scale, because recursive long-horizon optical flow can "
                        "become singular. dtw_m_mean is the "
                        "minimum-total-cost DTW path divided by its path length; "
                        "dtw_m_sum is retained for formula-level inspection."
                    ),
                }
            )
        fidelity[label] = fidelity_entry

        mass_kg = masses.get(label.strip().lower())
        if mass_kg is None:
            mechanics[label] = {
                "status": "pending",
                "note": "no mass entry for this label in kin_unity_object_masses_json",
            }
            continue

        # Planar Table-4 scenes already have a stronger metric reconstruction
        # than monocular depth: every tracked image point is intersected with
        # the calibrated Unity motion plane using the timestamp-matched Unity
        # camera. Use those metre-valued positions directly for Eq. 11--13.
        # This is both deterministic and avoids depth-domain noise.
        if len(metric_frames) >= 2:
            traj_world_sorted = sorted(traj_world, key=lambda item: item[0])
            peak_frame_gt = traj_world_sorted[0][0] if traj_world_sorted else None
            peak_speed = 0.0
            for (f0, x0), (f1, x1) in zip(traj_world_sorted, traj_world_sorted[1:]):
                dt_pair = (f1 - f0) / fps if fps > 0 else 1.0
                if dt_pair <= 0:
                    continue
                speed = float(np.linalg.norm(x1[[0, 2]] - x0[[0, 2]])) / dt_pair
                if speed > peak_speed:
                    peak_speed = speed
                    peak_frame_gt = f1
            ref_momentum = mass_kg * peak_speed
            ref_energy = 0.5 * mass_kg * peak_speed ** 2
            world_xz_by_frame = {
                frame: position[[0, 2]] for frame, position in traj_world
            }

            def planar_stage(frames_window: List[int]) -> Optional[Dict[str, Any]]:
                usable = [
                    frame for frame in frames_window
                    if frame in candidate_xz and frame in world_xz_by_frame
                ]
                if len(usable) < 2:
                    return None
                hat_pts = np.stack([candidate_xz[frame] for frame in usable])
                gt_pts = np.stack([world_xz_by_frame[frame] for frame in usable])
                return _mechanics_values_from_points(
                    label, mass_kg, usable, hat_pts, gt_pts, fps,
                    ref_momentum, ref_energy, peak_speed,
                )

            stages: Dict[str, Any] = {}
            if mechanics_estimator == "aligned_peak_finite_difference":
                aligned_peak_result = _aligned_peak_event_stage(
                    label,
                    mass_kg,
                    candidate_xz,
                    world_xz_by_frame,
                    fps,
                    tolerance_frames=event_alignment_tolerance_frames,
                    min_neighbor_ratio=event_min_neighbor_ratio,
                )
                if aligned_peak_result:
                    stages["aligned_peak_event"] = aligned_peak_result
            terminal_window = metric_frames[-depth_lift_window:]
            terminal_result = planar_stage(terminal_window)
            if terminal_result:
                stages["terminal"] = terminal_result
            peak_motion_same_as_terminal = False
            if peak_frame_gt is not None:
                peak_window = sorted(
                    sorted(metric_frames, key=lambda frame: abs(frame - peak_frame_gt))[
                        :depth_lift_window
                    ]
                )
                if peak_window != terminal_window:
                    peak_result = planar_stage(peak_window)
                    if peak_result:
                        stages["peak_motion"] = peak_result
                elif terminal_result:
                    peak_motion_same_as_terminal = True
            if stages:
                mechanics[label] = {
                    "status": "metric_planar_multistage",
                    "position_source": metric_mapping_source,
                    "stage_selection_for_summary": (
                        "aligned_peak_event_then_peak_motion_then_terminal"
                        if mechanics_estimator == "aligned_peak_finite_difference"
                        else "peak_motion_preferred_then_terminal"
                    ),
                    "reference_scale": {
                        "peak_speed_m_s": round(peak_speed, 4),
                        "peak_frame": int(peak_frame_gt) if peak_frame_gt is not None else None,
                        "reference_momentum_kg_m_s": round(ref_momentum, 4),
                        "reference_energy_j": round(ref_energy, 4),
                    },
                    "stages": stages,
                    "peak_motion_same_as_terminal": peak_motion_same_as_terminal,
                    "mass_kg": mass_kg,
                    "note": (
                        "Paper Eq. 11--13 computed from calibrated planar metre "
                        "positions and simulator-provided mass. When enabled, the "
                        "summary uses a bounded event-aligned one-step finite "
                        "difference (the paper's velocity definition), with a "
                        "persistence gate against isolated tracking spikes. The "
                        "legacy peak window and terminal stage remain auditable "
                        "fallbacks."
                    ),
                }
                continue

        if depth_lift_enabled and len(common_frames) >= 2:
            world_by_frame = {f: pos for f, pos in traj_world}

            # Reference scale: this object's own peak speed across its FULL
            # ground-truth trajectory (not just the evaluation window). The
            # paper's Eq 13 divides by the ground-truth value AT THE INSTANT
            # BEING EVALUATED, which is fine when that instant has real
            # motion, but explodes for a near-motionless instant regardless
            # of how small the absolute error is (e.g. an object that's
            # never hit: BilliardGame's stationary "red ball" produced
            # error_rate_pct in the tens of thousands of percent). That's a
            # real, fixable weakness of evaluating only one instant, not
            # something inherent to momentum/energy error -- normalizing
            # against the object's own peak momentum/energy instead gives a
            # stable, physically meaningful scale, and correctly degrades to
            # "no reference scale" (both peak values ~0) for an object that
            # truly never moves, rather than fabricating a number.
            traj_world_sorted = sorted(traj_world, key=lambda t: t[0])
            peak_frame_gt = traj_world_sorted[0][0] if traj_world_sorted else None
            peak_speed = 0.0
            for (f0, x0), (f1, x1) in zip(traj_world_sorted, traj_world_sorted[1:]):
                dt_pair = (f1 - f0) / fps if fps > 0 else 1.0
                if dt_pair <= 0:
                    continue
                speed = float(np.linalg.norm(x1 - x0)) / dt_pair
                if speed > peak_speed:
                    peak_speed = speed
                    peak_frame_gt = f1
            ref_momentum = mass_kg * peak_speed
            ref_energy = 0.5 * mass_kg * peak_speed ** 2

            stages: Dict[str, Any] = {}
            terminal_window = common_frames[-depth_lift_window:]
            terminal_result = _mechanics_stage(
                label, mass_kg, terminal_window, cand_by_frame, world_by_frame,
                metric_points_cap, intrinsics, depth_pipe, fps, ref_momentum, ref_energy, peak_speed,
            )
            if terminal_result:
                stages["terminal"] = terminal_result

            # Second stage, not just a terminal-window snapshot: evaluate
            # around this object's own peak-ground-truth-speed frame too
            # (paper Eq 11-13 gives "terminal momentum magnitude" only as
            # an *example* metric, not the sole prescribed instant -- for
            # an object whose real dynamics happen mid-clip, evaluating
            # only at the end misses the actual moment of interest).
            peak_motion_same_as_terminal = False
            if peak_frame_gt is not None:
                peak_window = sorted(
                    sorted(common_frames, key=lambda f: abs(f - peak_frame_gt))[:depth_lift_window]
                )
                if peak_window != terminal_window:
                    peak_result = _mechanics_stage(
                        label, mass_kg, peak_window, cand_by_frame, world_by_frame,
                        metric_points_cap, intrinsics, depth_pipe, fps, ref_momentum, ref_energy, peak_speed,
                    )
                    if peak_result:
                        stages["peak_motion"] = peak_result
                elif terminal_result:
                    # Every `stages[...]` entry is always a dict (never a
                    # bare string) so a consumer can uniformly do
                    # stages[stage_name]["momentum_kg_m_s"] without a type
                    # check -- when the peak and terminal windows coincide
                    # (e.g. a near-static object, or one whose motion peaks
                    # right at the tracked clip's end), skip the redundant
                    # duplicate stage entirely and record that fact as a
                    # sibling flag instead.
                    peak_motion_same_as_terminal = True

            if stages:
                mechanics[label] = {
                    "status": "metric_depth_lifted_multistage",
                    "note": (
                        "Momentum/kinetic-energy (paper Eq 11-13) via monocular-depth-"
                        "lifted velocity (linear-fit slope across a trailing window of "
                        "tracked frames, not a 2-point finite difference -- far more "
                        "sensitive to per-frame depth-map noise). Reported at two "
                        "stages: 'terminal' (last tracked frames, the paper's own "
                        "example instant) and 'peak_motion' (frames nearest this "
                        "object's own fastest ground-truth instant, extending beyond "
                        "the paper's literal example since evaluating only the "
                        "terminal instant can miss where the real dynamics happened). "
                        "Each stage reports error_rate_pct (Eq 13 exactly, divides by "
                        "|q_gt|) AND error_rate_pct_normalized (divides by this "
                        "object's own peak momentum/energy across its FULL trajectory "
                        "instead -- stays interpretable even when the evaluated "
                        "instant's q_gt is near zero). Still inherits the depth "
                        "model's own accuracy limits -- see DEPTH_DEFAULT_MODEL's "
                        "docstring for the metric-vs-relative and indoor-vs-outdoor "
                        "checkpoint caveats."
                    ),
                    "reference_scale": {
                        "peak_speed_m_s": round(peak_speed, 4),
                        "peak_frame": int(peak_frame_gt) if peak_frame_gt is not None else None,
                        "reference_momentum_kg_m_s": round(ref_momentum, 4),
                        "reference_energy_j": round(ref_energy, 4),
                        "note": (
                            f"peak_speed_m_s never exceeds {NEAR_ZERO_SPEED_M_S} m/s "
                            "-- this object never moves meaningfully in this rollout, so "
                            "error_rate_pct_normalized is reported as None below (see "
                            "each stage's error_rate_pct_normalized_note); only "
                            "absolute_errors is interpretable for such an object."
                            if peak_speed < NEAR_ZERO_SPEED_M_S else
                            "A real, physically meaningful scale for this object in "
                            "this rollout, used as the denominator for "
                            "error_rate_pct_normalized."
                        ),
                    },
                    "stages": stages,
                    "peak_motion_same_as_terminal": peak_motion_same_as_terminal,
                    "mass_kg": mass_kg,
                }
                continue

        v_hat_series = _velocity_series(traj_hat, dt)
        if len(v_hat_series) == 0:
            continue
        speed_hat = float(np.linalg.norm(v_hat_series[-1]))
        mechanics[label] = {
            "status": "pixel_units_only",
            "note": (
                "momentum/kinetic-energy require metric-scale velocity; depth-lifting "
                "was not enabled for this case (kin_depth_lift) or failed to produce a "
                "3D point at the terminal frames, so velocity here is in px/sec, not "
                "m/s, and p_hat/E_hat cannot be compared to Unity's SI-unit "
                "references. Reporting the raw terminal pixel speed for inspection."
            ),
            "estimated_terminal_speed_px_per_sec": round(speed_hat, 3),
            "mass_kg": mass_kg,
        }

    if metric_points_cap is not None:
        metric_points_cap.release()

    result["trajectory_fidelity"] = fidelity if fidelity else None
    normalized_tracked = {label.strip().lower() for label in trajectories_out}
    scored_labels = {
        label.strip().lower()
        for label, entry in fidelity.items()
        if isinstance(entry, dict)
        and any(
            isinstance(entry.get(metric), (int, float))
            for metric in ("fde_px", "dtw_px", "fde_m", "dtw_m_mean")
        )
    }
    missing_labels = [
        label for label in required_labels if label.lower() not in scored_labels
    ]
    result["trajectory_coverage"] = {
        "status": "complete" if not missing_labels else "incomplete_required_labels",
        "required_labels": required_labels,
        "tracked_labels": sorted(trajectories_out),
        "scored_labels": sorted(fidelity),
        "missing_required_labels": missing_labels,
        "required_label_count": len(required_labels),
        "tracked_required_label_count": sum(
            label.lower() in normalized_tracked for label in required_labels
        ),
        "scored_required_label_count": len(required_labels) - len(missing_labels),
        "missing_required_label_count": len(missing_labels),
    }
    result["mechanical_quantities"] = mechanics if mechanics else None
    result["mechanical_summary"] = _summarize_mechanical_quantities(mechanics)
    # Non-headline additions (candidate-camera FDE/DTW, robust mechanics);
    # computed after, and independently of, every legacy field above.
    try:
        result["extended_kinematics"] = extended_kinematics_from_tracks(
            trajectories_out,
            fps=fps,
            track_point=track_point,
            video_path=video_path,
            gt=gt,
            poses=poses,
            intrinsics=intrinsics,
            plane_calibration=loaded_plane_calibration,
            masses=masses,
            required_labels=required_labels,
            alignment_tolerance_frames=event_alignment_tolerance_frames,
            ego_params=_prefixed_params(artifacts, "kin_ego_"),
            robust_params=_prefixed_params(artifacts, "kin_robust_"),
            reference_video=getattr(context, "reference_video", None),
        )
    except Exception as exc:  # never let the additions break legacy output
        result["extended_kinematics"] = {"status": "error", "error": str(exc), "summary": None}
    # Ego-vehicle path in road coordinates (evaluation/modules/road_trajectory.py): a chase
    # camera keeps the car centred, so the car's pixel track cannot show where it went.
    if artifacts.get("kin_road_trajectory") and plane_calibration_path:
        try:
            from . import road_trajectory
            result["road_trajectory"] = road_trajectory.score_video(
                str(video_path), str(gt_json_path), str(poses_path), str(intrinsics_path),
                str(plane_calibration_path))
        except Exception as exc:
            result["road_trajectory"] = {"status": "error", "error": str(exc), "fde_road_m": None}
    return result


def _prefixed_params(artifacts: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """``kin_ego_<key>`` / ``kin_robust_<key>`` case overrides of the
    kinematics_robust defaults."""
    return {
        key[len(prefix):]: value
        for key, value in (artifacts or {}).items()
        if isinstance(key, str) and key.startswith(prefix)
    }


# ---------------------------------------------------------------------------
# Extended (non-headline) kinematics: candidate-camera lifting + robust
# mechanics. See evaluation/modules/kinematics_robust.py.
# ---------------------------------------------------------------------------
def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(float(v))]
    return round(float(np.mean(finite)), 6) if finite else None


def _reference_object_boxes(
    trajectories: Dict[str, List[Dict[str, Any]]],
    gt: Dict[str, Any],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    size: Tuple[int, int],
) -> Dict[int, List[List[float]]]:
    """Object masks for the Unity reference video: each tracked label's Unity
    position projected through the Unity camera, boxed with the candidate's
    median box size (bottom edge at the projected ground point)."""
    from . import kinematics_robust as kr

    boxes: Dict[int, List[List[float]]] = {}
    for label, points in trajectories.items():
        world = _unity_trajectory(gt, label)
        if not world:
            continue
        sizes = [
            (b[2] - b[0], b[3] - b[1])
            for b in kr.object_boxes_from_trajectory(points).values()
        ]
        w, h = np.median(np.asarray(sizes), axis=0) if sizes else (0.15 * size[0], 0.2 * size[1])
        for frame, (u, v) in _project_unity_trajectory_to_pixels(world, poses, intrinsics).items():
            boxes.setdefault(int(frame), []).append([u - w / 2, v - h, u + w / 2, v + 0.3 * h])
    return boxes


def extended_kinematics_from_tracks(
    trajectories: Dict[str, List[Dict[str, Any]]],
    *,
    fps: float,
    track_point: str,
    video_path: Optional[str],
    gt: Dict[str, Any],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    plane_calibration: Optional[Dict[str, Any]],
    masses: Dict[str, float],
    required_labels: Sequence[str],
    alignment_tolerance_frames: int,
    ego_params: Optional[Dict[str, Any]] = None,
    robust_params: Optional[Dict[str, Any]] = None,
    reference_video: Optional[str] = None,
) -> Dict[str, Any]:
    """Candidate-camera FDE/DTW and robust mechanics from tracked pixels.

    Pure function of the tracked trajectories (``result["trajectories"]``),
    the candidate video (for its camera motion) and the Unity artifacts, so
    stored results can be re-scored without re-running the detector
    (``tools/paper_tables/rescore_kinematics_extended.py``).  Legacy fields are untouched.
    """
    from . import kinematics_robust as kr

    camera = kr.unity_camera_motion(poses)
    out: Dict[str, Any] = {
        "schema_version": 1,
        "unity_camera": camera,
        "labels": {},
    }
    if plane_calibration is None:
        out["status"] = "not_configured"
        out["note"] = "kin_plane_calibration_json is required for metric lifting"
        out["summary"] = None
        return out
    plane_y = float(plane_calibration["plane"]["value_m"])
    reliable = True
    reasons: List[str] = []
    transforms: Dict[int, np.ndarray] = {}
    if camera["moving"]:
        out["lifting"] = "candidate_camera_ground_plane"
        if not video_path:
            out["status"] = "error"
            out["error"] = "candidate video is required for candidate-camera lifting"
            out["summary"] = None
            return out
        cap = cv2.VideoCapture(str(video_path))
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        h0, anchor_rms = kr.frame0_plane_homography(plane_calibration, size)
        boxes: Dict[int, List[List[float]]] = {}
        for points in trajectories.values():
            for frame, box in kr.object_boxes_from_trajectory(points).items():
                boxes.setdefault(frame, []).append(box)
        motion_params = {**(ego_params or {}), "motion_model": kr.resolve_motion_model(camera, ego_params)}
        transforms, diagnostics = kr.estimate_plane_camera_motion(
            str(video_path), h0, fps=fps, object_boxes=boxes, params=motion_params
        )
        if (motion_params.get("split_check", kr.EGO_DEFAULTS["split_check"])):
            diagnostics["split_consistency"] = kr.split_consistency(
                str(video_path), h0, transforms, np.asarray(poses[0][[0, 2], 3], dtype=np.float64),
                fps=fps, object_boxes=boxes, params=motion_params,
            )
        floor = None
        if reference_video:
            try:
                floor = kr.reference_camera_floor(
                    str(reference_video),
                    h0,
                    poses,
                    object_boxes=_reference_object_boxes(trajectories, gt, poses, intrinsics, size),
                    params=motion_params,
                )
            except Exception as exc:  # the floor is a gate, never a crash
                floor = {"error": str(exc), "relative_max_error": None}
            out["reference_camera_floor"] = floor
        reliable, reasons = kr.ego_reliability(diagnostics, ego_params, floor)
        diagnostics["anchor_fit_rms_m"] = round(anchor_rms, 6)
        diagnostics["reliable"] = reliable
        diagnostics["unreliable_reasons"] = reasons
        out["camera_motion_estimate"] = diagnostics
    else:
        out["lifting"] = "unity_camera_static"
    out["reliable"] = reliable

    label_rows: Dict[str, Any] = {}
    for label, points in trajectories.items():
        traj_world = _unity_trajectory(gt, label)
        if traj_world is None or len(traj_world) < 2:
            continue
        pixels = {
            int(p["frame"]): np.asarray(
                p["bbox_bottom_center_uv"] if track_point == "bbox_bottom_center" else p["centroid_uv"],
                dtype=np.float64,
            )
            for p in points
        }
        reference_xz = {int(f): pos[[0, 2]] for f, pos in traj_world}
        row: Dict[str, Any] = {}
        unity_lifted = _map_pixels_to_unity_plane_by_pose(pixels, poses, intrinsics, plane_y)
        mechanics_points: Optional[Dict[int, np.ndarray]] = None
        if camera["moving"]:
            lifted = kr.lift_points(pixels, transforms)
            lifted, anchor = kr.anchor_contact_offset(lifted, reference_xz, transforms, ego_params)
            row["contact_offset_anchor"] = anchor
            attachment = kr.object_camera_attachment(traj_world, poses, intrinsics, float(2.0 * intrinsics[0][2]))
            row["object_camera_attachment"] = attachment
            if reliable:
                mechanics_points, row["mechanics_position_source"] = lifted, "candidate_camera"
            elif not attachment["attached"]:
                # Velocities of an object the camera does not carry stay
                # observable through the timestamp-matched Unity camera.
                mechanics_points = unity_lifted
                row["mechanics_position_source"] = "unity_camera_fallback_unattached_object"
            else:
                row["mechanics_position_source"] = "withheld_unreliable_camera_attached_object"
        else:
            lifted = unity_lifted
            mechanics_points, row["mechanics_position_source"] = lifted, "unity_camera_static"
        frames = sorted(set(lifted) & set(reference_xz))
        row["metric_frames_compared"] = len(frames)
        if len(frames) >= 2:
            hat = np.stack([lifted[f] for f in frames])
            ref = np.stack([reference_xz[f] for f in frames])
            row["fde_m_ego"] = round(_fde(hat, ref), 6)
            row["dtw_m_ego"] = round(_dtw_mean(hat, ref), 6)
        else:
            row["status"] = "insufficient_overlap"
        if mechanics_points is not None:
            # Far-field guard: near the plane's horizon a pixel spans metres,
            # so a one-pixel detector jitter becomes a huge speed.
            shifted = {f: p + np.asarray([0.0, 1.0]) for f, p in pixels.items()}
            if row.get("mechanics_position_source") == "candidate_camera":
                base, step = kr.lift_points(pixels, transforms), kr.lift_points(shifted, transforms)
            else:
                base = unity_lifted
                step = _map_pixels_to_unity_plane_by_pose(shifted, poses, intrinsics, plane_y)
            max_scale = float({**kr.ROBUST_DEFAULTS, **(robust_params or {})}["max_m_per_px"])
            far = {
                f for f in base
                if f in step and float(np.linalg.norm(step[f] - base[f])) > max_scale
            }
            far |= set(mechanics_points) - set(base)
            if far:
                row["mechanics_far_field_frames_dropped"] = len(far)
                mechanics_points = {f: p for f, p in mechanics_points.items() if f not in far}
        mass_kg = masses.get(label.strip().lower())
        if mass_kg is not None and mechanics_points is not None and len(frames) >= 2:
            mechanics = kr.robust_mechanics(
                mechanics_points,
                reference_xz,
                fps,
                mass_kg=float(mass_kg),
                alignment_tolerance_frames=alignment_tolerance_frames,
                params=robust_params,
            )
            row["robust_mechanics"] = mechanics
        label_rows[label] = row
    out["labels"] = label_rows

    scored = {
        label.strip().lower()
        for label, row in label_rows.items()
        if isinstance(row.get("fde_m_ego"), (int, float))
    }
    missing = [label for label in required_labels if label.strip().lower() not in scored]
    rows = list(label_rows.values())
    mech = [
        row["robust_mechanics"]
        for row in rows
        if isinstance(row.get("robust_mechanics"), dict)
        and row["robust_mechanics"].get("status") == "computed"
    ]
    publish_fidelity = reliable and not missing
    summary = {
        "lifting": out["lifting"],
        "reliable": reliable,
        "missing_required_labels": missing,
        # withheld (None) when the camera estimate is unreliable or a required
        # label is untracked; the per-label raw values stay inspectable above
        "fde_m_ego_mean": _mean_or_none([r.get("fde_m_ego") for r in rows]) if publish_fidelity else None,
        "dtw_m_ego_mean": _mean_or_none([r.get("dtw_m_ego") for r in rows]) if publish_fidelity else None,
    }
    for key in (
        "robust_error_rate_pct",
        "robust_momentum_error_pct",
        "robust_energy_error_pct",
        "peak_speed_error_pct",
        "contact_dp_error_pct",
    ):
        summary[key] = _mean_or_none([m.get(key) for m in mech])
    out["summary"] = summary
    out["status"] = "computed"
    return out
