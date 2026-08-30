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
    v_hat = _linear_velocity_fit(usable, hat_pts, fps)
    v_gt = _linear_velocity_fit(usable, gt_pts, fps)
    p_hat = mass_kg * float(np.linalg.norm(v_hat))
    p_gt = mass_kg * float(np.linalg.norm(v_gt))
    e_hat = 0.5 * mass_kg * float(np.linalg.norm(v_hat)) ** 2
    e_gt = 0.5 * mass_kg * float(np.linalg.norm(v_gt)) ** 2

    # Near-zero baseline thresholds, derived from the ONE physical speed
    # constant (NEAR_ZERO_SPEED_M_S) through this object's own real mass --
    # not independently-chosen absolute values in momentum/energy units.
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
        "frames_used": [int(f) for f in usable],
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
    color_filter_enabled = bool(artifacts.get("kin_color_filter", False))
    color_hue_tolerance_deg = float(artifacts.get("kin_color_hue_tolerance_deg", 25.0))

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
    result["mechanical_quantities"] = mechanics if mechanics else None
    return result
