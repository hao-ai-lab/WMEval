"""Candidate-camera lifting and robust mechanics for kinematics (non-headline).

Two additions to :mod:`evaluation.modules.kinematics_mechanics`, reported
beside -- never instead of -- the paper-reconstruction metrics:

1. **Candidate-camera lifting** (``fde_m_ego`` / ``dtw_m_ego``).  The legacy
   metric lifts candidate pixels to Unity X/Z metres through the *Unity*
   reference camera at the same timestamp.  With a camera that follows the
   tracked object (RacingCar's third-person chase camera, BilliardGame's
   camera riding on the cue ball) the object stays near the same pixel in any
   video, so that lifting returns Unity's own path whatever the candidate
   does.  For cases whose Unity camera moves, this module estimates the
   candidate video's *own* camera motion relative to the ground plane:
   ground-plane features are tracked frame to frame (pyramidal LK, masked to
   below the plane's horizon and away from the tracked object), a RANSAC
   homography to the most recent keyframe is fitted per frame, and keyframes
   (every ``keyframe_interval_sec``) chain back to frame 0.  Frame 0 is the
   conditioning frame shared with Unity, so the plane calibration's frame-0
   anchors (pixel <-> world X/Z) fix metric scale::

       G_t = H0 @ A_k @ H_{t->k}     (frame-t pixel -> world X/Z metres)

   The metric chain is constrained to the ground-plane rigid motion of a
   camera mounted at constant height/pitch above the local ground
   (``ground_rigid``; ``ground_translation`` when the Unity camera never
   rotates), which removes the scale/tilt drift of a free homography chain on
   non-planar roads.  Static-camera cases keep the legacy Unity-camera
   lifting (identical values, reported under the same new names so one
   column covers every case).  Each row carries drift/quality diagnostics
   (keyframe reprojection error, inliers, left/right split consistency, and
   the case-level floor on the Unity reference); rows failing a gate are
   reported with ``reliable: false`` and their ``*_ego`` values withheld from
   the flat metrics instead of silently scored.

2. **Robust mechanics** (``robust_*``).  Paper Eq. 12/13 evaluate momentum and
   kinetic energy from a one-step finite difference at a single aligned peak
   instant and divide by that instant's value; tracking jitter is amplified
   by the difference and the denominator can be near zero, so a reference
   scored against itself is no better than a wrong trajectory.  Here both the
   candidate and the Unity trajectory (resampled on the candidate's observed
   frames) are Savitzky--Golay smoothed and differentiated, the candidate is
   shifted by the bounded motion-onset offset (within the module's event
   alignment tolerance), and over the span where the GT speed exceeds a
   threshold we report:

   * ``robust_momentum_error_pct`` -- mean |p_hat(t) - p(t)| / p_peak;
   * ``robust_energy_error_pct``   -- mean |E_hat(t) - E(t)| / E_peak;
   * ``robust_error_rate_pct``     -- mean of the two (the robust analogue of
     the paper's Error Rate column);
   * ``peak_speed_error_pct``      -- |max v_hat - max v| / max v;
   * ``contact_dp_error_pct``      -- for impulsive GT events, the momentum
     change across the contact from linear fits before/after it,
     |dp_hat - dp| / |dp| (vector).

   Eq. 12/13 values stay in the legacy fields unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

EPS = 1e-9

# ---------------------------------------------------------------------------
# Defaults (calibrated on the Lite Unity references; see suite.yaml's dated
# 2026-10-21 kinematics note and docs/evaluation_dimensions.md)
# ---------------------------------------------------------------------------
# A Unity camera whose per-frame centre moves by more than this (std of the
# X/Y/Z translation, metres) or yaws/pitches by more than MOVING_CAMERA_ROT_DEG
# is a moving camera; candidate-camera lifting applies.
MOVING_CAMERA_TRANSLATION_STD_M = 0.05
MOVING_CAMERA_ROT_DEG = 2.0

EGO_DEFAULTS: Dict[str, Any] = {
    # "auto": ground_rigid when the Unity camera rotates, else
    # ground_translation (resolved by resolve_motion_model)
    "motion_model": "auto",
    "keyframe_interval_sec": 0.5,
    "ransac_reproj_px": 2.0,  # at 1280 px width; scaled with the video
    "tracker": "lk",  # "lk" (pyramidal Lucas-Kanade) or "dis" (dense DIS flow)
    "max_corners": 600,
    "feature_quality": 0.005,
    "fb_max_px": 1.0,
    "min_inliers": 15,
    "anchor_contact_offset": True,
    "anchor_max_frame": 3,
    "horizon_margin_frac": 0.22,
    "object_mask_pad_frac": 0.35,
    # Reliability gates (row is reported but flagged unreliable above them)
    "max_keyframe_reproj_px": 3.0,  # median per-keyframe reprojection error
    "min_inliers_median": 20,
    # left/right split consistency (row level): max(max_split_m,
    # max_split_relative * estimated camera path)
    "split_check": True,
    "max_split_m": 0.75,
    "max_split_relative": 0.05,
    # case level: the estimator on the Unity reference must follow Unity's own
    # camera within this fraction of the camera path
    "max_reference_relative_error": 0.05,
    "min_coverage": 0.9,
    "max_failed_frame_frac": 0.05,
}

ROBUST_DEFAULTS: Dict[str, Any] = {
    "smoothing_window_sec": 0.27,
    "polyorder": 2,
    "moving_speed_frac": 0.3,
    "moving_speed_floor_m_s": 0.05,
    "min_span_coverage": 0.5,
    # mechanics samples whose lifted pixel spans more than this many metres
    # (near the plane's horizon) are dropped
    "max_m_per_px": 0.1,
    "contact_accel_frac": 4.0,  # x peak_speed / s: impulsive acceleration
    "contact_fit_window_sec": 0.2,
    "contact_guard_sec": 0.07,
    "contact_min_dv_frac": 0.25,
    "contact_impulsive_ratio": 2.0,
}


# ---------------------------------------------------------------------------
# Camera-motion classification
# ---------------------------------------------------------------------------
def unity_camera_motion(poses: np.ndarray) -> Dict[str, Any]:
    """Classify the Unity reference camera as static or moving."""
    poses = np.asarray(poses, dtype=np.float64)
    centres = poses[:, :3, 3]
    forward = poses[:, :3, 2]
    yaw = np.degrees(np.unwrap(np.arctan2(forward[:, 0], forward[:, 2])))
    pitch = np.degrees(np.arcsin(np.clip(forward[:, 1], -1.0, 1.0)))
    translation_std = float(np.max(np.std(centres, axis=0))) if len(centres) else 0.0
    rotation_range = float(max(np.ptp(yaw), np.ptp(pitch))) if len(poses) else 0.0
    moving = (
        translation_std > MOVING_CAMERA_TRANSLATION_STD_M
        or rotation_range > MOVING_CAMERA_ROT_DEG
    )
    return {
        "moving": bool(moving),
        "translation_std_m": round(translation_std, 4),
        "rotation_range_deg": round(rotation_range, 3),
        "path_length_m": round(
            float(np.sum(np.linalg.norm(np.diff(centres[:, [0, 2]], axis=0), axis=1)))
            if len(centres) > 1 else 0.0,
            3,
        ),
    }


def object_camera_attachment(
    traj_world: Sequence[Tuple[int, np.ndarray]],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    image_width: float,
    spread_threshold_frac: float = 0.05,
) -> Dict[str, Any]:
    """Is the tracked object carried with the Unity camera (a chase camera)?

    The timestamp-matched Unity-camera lifting is blind exactly when the
    object's image position barely changes while it moves in the world: it
    then returns Unity's own path whatever the candidate does.  Measured on
    the Unity projection: attached when the object moves > 0.5 m but its
    projected pixel spreads (std) by < ``spread_threshold_frac`` of the image
    width.  Such an object's mechanics may never fall back to that lifting.
    """
    poses = np.asarray(poses, dtype=np.float64)
    world = np.stack([np.asarray(p, dtype=np.float64) for _, p in traj_world])
    path = float(np.sum(np.linalg.norm(np.diff(world[:, [0, 2]], axis=0), axis=1))) if len(world) > 1 else 0.0
    pixels = []
    for f, p in traj_world:
        pose = poses[min(max(int(f), 0), len(poses) - 1)]
        local = pose[:3, :3].T @ (np.asarray(p, dtype=np.float64) - pose[:3, 3])
        if local[2] <= EPS:
            continue
        fx, fy, cx, cy = intrinsics[min(max(int(f), 0), len(intrinsics) - 1)]
        pixels.append([fx * local[0] / local[2] + cx, fy * local[1] / local[2] + cy])
    pixels = np.asarray(pixels)
    spread = float(np.max(np.std(pixels, axis=0))) if len(pixels) > 1 else 0.0
    return {
        "attached": bool(path > 0.5 and spread < spread_threshold_frac * float(image_width)),
        "projected_pixel_spread_px": round(spread, 2),
        "world_path_m": round(path, 4),
    }


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------
def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    mapped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return mapped[:, :2] / mapped[:, 2:3]


def frame0_plane_homography(
    calibration: Mapping[str, Any], video_size: Tuple[int, int]
) -> Tuple[np.ndarray, float]:
    """Frame-0 pixel -> world X/Z from the calibration anchors (least squares).

    Returns the homography and the anchors' RMS fit residual in metres.
    """
    source = calibration.get("image_size") or list(video_size)
    scale = np.asarray([video_size[0] / float(source[0]), video_size[1] / float(source[1])])
    anchors = calibration["anchors"]
    uv = np.asarray([a["conditioning_uv"] for a in anchors], dtype=np.float64) * scale
    xz = np.asarray([a["world_xz_m"] for a in anchors], dtype=np.float64)
    if len(uv) < 4:
        raise ValueError("plane calibration needs at least four anchors")
    matrix, _ = cv2.findHomography(uv, xz, 0)
    if matrix is None:
        raise ValueError("plane calibration anchors are degenerate")
    residual = apply_homography(matrix, uv) - xz
    return matrix, float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


def ground_side_mask(
    plane_from_pixel: np.ndarray,
    shape: Tuple[int, int],
    margin_frac: float,
) -> np.ndarray:
    """uint8 mask of pixels on the ground side of the plane's horizon.

    The horizon of frame t is the line where the third row of
    ``plane_from_pixel`` vanishes; the bottom-centre pixel is taken to be on
    the ground (true for every Lite kinematics view).  Pixels within
    ``margin_frac`` of the image height of the horizon are excluded too: depth
    diverges there and off-plane structure (trees, walls) dominates.
    """
    height, width = shape
    row = np.asarray(plane_from_pixel, dtype=np.float64)[2]
    sign = np.sign(row @ np.asarray([width / 2.0, height - 1.0, 1.0])) or 1.0
    vs, us = np.mgrid[0:height, 0:width]
    w = sign * (row[0] * us + row[1] * vs + row[2])
    mask = (w > 0).astype(np.uint8) * 255
    margin = int(round(margin_frac * height))
    if margin > 0 and np.any(mask == 0):
        # erode upward from the horizon: shift the "sky" region down by margin
        sky = mask == 0
        shifted = np.zeros_like(sky)
        shifted[margin:] = sky[:-margin]
        mask[shifted | sky] = 0
    border = max(4, min(width, height) // 80)
    mask[:border] = 0
    mask[-border:] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask


def _object_box_mask(mask: np.ndarray, box: Any, pad_frac: float) -> None:
    """Zero the (padded) object box(es) in ``mask``; ``box`` is one
    ``[x1, y1, x2, y2]`` or a list of them."""
    if box is None or len(box) == 0:
        return
    if isinstance(box[0], (list, tuple, np.ndarray)):
        for item in box:
            _object_box_mask(mask, item, pad_frac)
        return
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    pad_x, pad_y = pad_frac * w, pad_frac * h
    # extend further downward/sideways for the cast shadow
    x1, x2 = x1 - pad_x, x2 + pad_x
    y1, y2 = y1 - pad_y, y2 + 2.0 * pad_y
    height, width = mask.shape
    cv2.rectangle(
        mask,
        (int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))),
        (int(min(width - 1, np.ceil(x2))), int(min(height - 1, np.ceil(y2)))),
        0,
        -1,
    )


def object_boxes_from_trajectory(points: Sequence[Mapping[str, Any]]) -> Dict[int, List[float]]:
    """Per-frame object box; reconstructed from centroid/bottom when absent.

    Older stored results carry only ``centroid_uv`` and ``bbox_bottom_center_uv``;
    the box height is then twice their vertical gap and its width is taken as
    twice the height (generous, it is only a feature mask).
    """
    out: Dict[int, List[float]] = {}
    for p in points:
        box = p.get("bbox_xyxy")
        if box is not None and len(box) == 4:
            out[int(p["frame"])] = [float(v) for v in box]
            continue
        centroid = p.get("centroid_uv")
        bottom = p.get("bbox_bottom_center_uv")
        if centroid is None or bottom is None:
            continue
        half_h = max(abs(float(bottom[1]) - float(centroid[1])), 4.0)
        cx, cy = float(centroid[0]), float(centroid[1])
        out[int(p["frame"])] = [cx - 2 * half_h, cy - half_h, cx + 2 * half_h, cy + half_h]
    return out


def _nearest_box(boxes: Mapping[int, Sequence[float]], frame: int, max_gap: int = 15):
    if not boxes:
        return None
    if frame in boxes:
        return boxes[frame]
    nearest = min(boxes, key=lambda k: abs(k - frame))
    return boxes[nearest] if abs(nearest - frame) <= max_gap else None


# ---------------------------------------------------------------------------
# Ground-plane camera-motion estimation
# ---------------------------------------------------------------------------
class _Frames:
    """Grayscale frames from a video path or an in-memory sequence."""

    def __init__(self, source: Any):
        self.source = source

    def __iter__(self):
        if isinstance(self.source, (str, bytes)) or hasattr(self.source, "__fspath__"):
            cap = cv2.VideoCapture(str(self.source))
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            finally:
                cap.release()
        else:
            for frame in self.source:
                frame = np.asarray(frame)
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield frame.astype(np.uint8)


def _sample_flow(flow: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, 2) flow field at (N, 2) pixel points."""
    pts = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    sampled = cv2.remap(
        flow, pts[..., 0], pts[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return sampled.reshape(-1, 2).astype(np.float64)


def _fit_homography(src: np.ndarray, dst: np.ndarray, thresh: float):
    if len(src) < 4:
        return None, None
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    matrix, inliers = cv2.findHomography(
        src.reshape(-1, 1, 2).astype(np.float64),
        dst.reshape(-1, 1, 2).astype(np.float64),
        method,
        thresh,
        maxIters=5000,
        confidence=0.999,
    )
    if matrix is None or inliers is None:
        return None, None
    return matrix, inliers.reshape(-1).astype(bool)


def _pixel_scale_weights(plane_from_pixel: np.ndarray, points: np.ndarray) -> np.ndarray:
    """1 / (metres per pixel)^2 of ``plane_from_pixel`` at each point.

    Ground points far from the camera move a fraction of a pixel per metre, so
    their pixel noise is a large metric error; weighting by the inverse
    squared local scale makes the fit an approximately pixel-noise-optimal one.
    """
    base = apply_homography(plane_from_pixel, points)
    du = apply_homography(plane_from_pixel, points + np.asarray([1.0, 0.0])) - base
    dv = apply_homography(plane_from_pixel, points + np.asarray([0.0, 1.0])) - base
    scale = 0.5 * (np.linalg.norm(du, axis=1) + np.linalg.norm(dv, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / np.square(scale)
    weights[~np.isfinite(weights)] = 0.0
    return weights


def fit_rigid_2d(
    source: np.ndarray, target: np.ndarray, weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """Weighted 2D Kabsch: 3x3 rigid T with target ~ T(source); and the
    weighted least-squares similarity scale (a diagnostic, not applied)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    w = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), EPS)
    mu_s = w @ source
    mu_t = w @ target
    s0 = source - mu_s
    t0 = target - mu_t
    cov = (s0 * w[:, None]).T @ t0
    u, sing, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rot = vt.T @ np.diag([1.0, d]) @ u.T
    var_s = float(np.sum(w * np.sum(s0 ** 2, axis=1)))
    scale = float((sing[0] + d * sing[1]) / var_s) if var_s > EPS else 1.0
    transform = np.eye(3)
    transform[:2, :2] = rot
    transform[:2, 2] = mu_t - rot @ mu_s
    return transform, scale


def estimate_plane_camera_motion(
    frames: Any,
    plane_from_pixel0: np.ndarray,
    *,
    fps: float,
    object_boxes: Optional[Mapping[int, Sequence[float]]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Chain the candidate video's ground-plane motion back to frame 0.

    ``frames`` is a video path or a sequence of BGR/gray images;
    ``plane_from_pixel0`` (H0) maps frame-0 pixels to world X/Z.  Returns
    ``{frame: G_t}`` (frame-t pixel -> world X/Z metres) and diagnostics.

    Per frame, ground features tracked from the current keyframe are fitted
    with a RANSAC homography (off-plane and moving points are rejected as
    outliers).  The metric chain then depends on ``motion_model``:

    * ``"ground_rigid"`` (default): the camera is rigidly mounted at a
      constant height and pitch above the *local* ground (a chase camera, a
      camera riding on a ball), so ``G_t = T_t @ H0`` with ``T_t`` a 2D rigid
      motion (yaw + X/Z translation).  Each step's increment is the weighted
      2D Kabsch fit between the inlier correspondences mapped through H0.
      Constraining the chain to 3 DOF removes the scale/tilt drift that a free
      8-DOF homography chain accumulates where the ground is not one plane
      (RacingCar's road climbs at ~3.4 %).
    * ``"ground_translation"``: as ``ground_rigid`` without yaw, for a camera
      that translates with a fixed heading (BilliardGame's cue-ball camera);
      chosen per case when the Unity camera never rotates.
    * ``"homography"``: the free chain ``G_t = H0 @ A_k @ H_{t->k}``.
    """
    cfg = {**EGO_DEFAULTS, **(params or {})}
    motion_model = str(cfg.get("motion_model", "ground_rigid"))
    if motion_model == "auto":
        motion_model = "ground_rigid"
    if motion_model not in {"ground_rigid", "ground_translation", "homography"}:
        raise ValueError("motion_model must be ground_rigid, ground_translation or homography")
    object_boxes = object_boxes or {}
    cv2.setRNGSeed(int(cfg.get("seed", 0)))  # RANSAC/USAC sampling: reproducible
    h0 = np.asarray(plane_from_pixel0, dtype=np.float64)
    keyframe_interval = max(1, int(round(float(cfg["keyframe_interval_sec"]) * max(fps, 1e-6))))
    lk = {
        "winSize": (21, 21),
        "maxLevel": 4,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    transforms: Dict[int, np.ndarray] = {}
    keyframe_errors: List[float] = []
    keyframe_world_errors: List[float] = []
    keyframe_scales: List[float] = []
    keyframe_inliers: List[int] = []
    keyframe_log: List[Dict[str, Any]] = []
    failed_frames: List[int] = []
    keyframes: List[int] = []

    previous_gray = None
    key_frame_index = 0
    key_metric = h0.copy()  # keyframe pixel -> world X/Z
    key_points = np.empty((0, 2), dtype=np.float32)
    track_points = np.empty((0, 2), dtype=np.float32)
    last_metric = h0.copy()  # current frame pixel -> world X/Z, last good fit
    last_step: Optional[Dict[str, Any]] = None
    thresh = float(cfg["ransac_reproj_px"])
    min_inliers = int(cfg["min_inliers"])
    n_frames = 0
    key_count = 0
    tracker = str(cfg.get("tracker", "lk"))
    if tracker not in {"lk", "dis"}:
        raise ValueError("tracker must be lk or dis")
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if tracker == "dis" else None

    def detect(gray: np.ndarray, frame_index: int, plane_from_pixel: np.ndarray) -> np.ndarray:
        mask = ground_side_mask(plane_from_pixel, gray.shape, float(cfg["horizon_margin_frac"]))
        _object_box_mask(mask, _nearest_box(object_boxes, frame_index), float(cfg["object_mask_pad_frac"]))
        region = str(cfg.get("region", "all"))
        if region == "left":
            mask[:, gray.shape[1] // 2:] = 0
        elif region == "right":
            mask[:, : gray.shape[1] // 2] = 0
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(cfg["max_corners"]),
            qualityLevel=float(cfg["feature_quality"]),
            minDistance=max(6, gray.shape[1] // 128),
            mask=mask,
            blockSize=7,
        )
        if corners is None:
            return np.empty((0, 2), dtype=np.float32)
        return corners.reshape(-1, 2).astype(np.float32)

    def metric_step(src: np.ndarray, dst: np.ndarray, homography: np.ndarray) -> Dict[str, Any]:
        """frame-t pixel -> world X/Z given inlier pairs (src in t, dst in key)."""
        if motion_model == "homography":
            metric = key_metric @ homography
            predicted = apply_homography(homography, src)
            return {"metric": metric, "predicted_key_px": predicted, "scale": 1.0}
        q_src = apply_homography(h0, src)
        q_dst = apply_homography(h0, dst)
        weights = _pixel_scale_weights(h0, dst.astype(np.float64))
        finite = np.all(np.isfinite(q_src), axis=1) & np.all(np.isfinite(q_dst), axis=1) & (weights > 0)
        h0_inv = np.linalg.inv(h0)
        robust = weights.copy()
        robust[~finite] = 0.0
        for _ in range(4):
            # IRLS with Cauchy weights on the pixel reprojection residual:
            # the rigid model is fitted to the dominant ground motion
            delta, scale = fit_rigid_2d(q_src[finite], q_dst[finite], robust[finite])
            if motion_model == "ground_translation":
                w_t = robust[finite] / max(float(robust[finite].sum()), EPS)
                delta = np.eye(3)
                delta[:2, 2] = w_t @ (q_dst[finite] - q_src[finite])
            predicted = apply_homography(h0_inv @ delta @ h0, src)
            residual = np.linalg.norm(predicted - dst, axis=1)
            residual[~np.isfinite(residual)] = np.inf
            robust = weights / (1.0 + np.square(residual / max(thresh, 1e-6)))
            robust[~finite] = 0.0
        # key_metric = T_k @ H0  ->  G_t = T_k @ delta @ H0
        metric = key_metric @ h0_inv @ delta @ h0
        return {"metric": metric, "predicted_key_px": predicted, "scale": scale}

    for frame_index, gray in enumerate(_Frames(frames)):
        n_frames += 1
        if previous_gray is None:
            thresh = float(cfg["ransac_reproj_px"]) * gray.shape[1] / 1280.0
            transforms[0] = h0.copy()
            keyframes.append(0)
            key_points = detect(gray, 0, h0)
            track_points = key_points.copy()
            key_count = len(key_points)
            previous_gray = gray
            continue

        ok_fit = False
        if len(track_points) >= 4:
            h, w = gray.shape
            if tracker == "dis":
                # dense DIS flow sampled at the features: steadier than
                # pyramidal LK on the low-contrast, temporally re-synthesised
                # textures of generated video
                flow_f = dis.calc(previous_gray, gray, None)
                flow_b = dis.calc(gray, previous_gray, None)
                fwd = track_points.reshape(-1, 2) + _sample_flow(flow_f, track_points.reshape(-1, 2))
                back = fwd + _sample_flow(flow_b, fwd)
                fb = np.linalg.norm(back - track_points.reshape(-1, 2), axis=1)
                status = np.ones(len(fwd), dtype=bool)
            else:
                forward, st_f, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray, gray, track_points.reshape(-1, 1, 2), None, **lk
                )
                backward, st_b, _ = cv2.calcOpticalFlowPyrLK(
                    gray, previous_gray, forward, None, **lk
                )
                fb = np.linalg.norm(backward.reshape(-1, 2) - track_points.reshape(-1, 2), axis=1)
                fwd = forward.reshape(-1, 2)
                status = st_f.reshape(-1).astype(bool) & st_b.reshape(-1).astype(bool)
            keep = (
                status
                & np.isfinite(fb)
                & (fb <= float(cfg["fb_max_px"]))
                & (fwd[:, 0] >= 0) & (fwd[:, 0] < w)
                & (fwd[:, 1] >= 0) & (fwd[:, 1] < h)
            )
            key_points = key_points[keep]
            track_points = fwd[keep].astype(np.float32)
            homography, inliers = _fit_homography(track_points, key_points, thresh)
            if homography is not None and int(inliers.sum()) >= min_inliers:
                # drop RANSAC outliers (off-plane / moving points) for good
                key_points = key_points[inliers]
                track_points = track_points[inliers]
                step = metric_step(track_points.astype(np.float64), key_points.astype(np.float64), homography)
                if np.all(np.isfinite(step["metric"])):
                    ok_fit = True
                    step["inliers"] = int(inliers.sum())
                    last_step = step
                    last_metric = step["metric"]
        if not ok_fit:
            failed_frames.append(frame_index)
        transforms[frame_index] = last_metric.copy()

        new_keyframe = (
            frame_index - key_frame_index >= keyframe_interval
            or len(track_points) < max(2 * min_inliers, 40, 0.5 * key_count)
            or not ok_fit
        )
        if new_keyframe:
            if ok_fit and last_step is not None and len(track_points):
                errors = np.linalg.norm(last_step["predicted_key_px"] - key_points, axis=1)
                errors = errors[np.isfinite(errors)]
                if len(errors):
                    keyframe_errors.append(float(np.median(errors)))
                    keyframe_log.append(
                        {
                            "frame": int(frame_index),
                            "reproj_px": round(float(np.median(errors)), 3),
                            "inliers": int(last_step["inliers"]),
                            "scale": round(float(last_step["scale"]), 4),
                        }
                    )
                keyframe_inliers.append(int(last_step["inliers"]))
                keyframe_scales.append(float(last_step["scale"]))
                # the same residual on the ground plane (metres): standard
                # error of this step's metric increment
                world_a = apply_homography(key_metric, key_points)
                world_b = apply_homography(key_metric, last_step["predicted_key_px"])
                d = np.linalg.norm(world_a - world_b, axis=1)
                d = d[np.isfinite(d)]
                if len(d):
                    keyframe_world_errors.append(float(np.median(d) / np.sqrt(len(d))))
            key_metric = last_metric.copy()
            key_frame_index = frame_index
            keyframes.append(frame_index)
            key_points = detect(gray, frame_index, key_metric)
            track_points = key_points.copy()
            key_count = len(key_points)
            last_step = None
        previous_gray = gray

    failed_frac = len(failed_frames) / max(n_frames - 1, 1)
    log_scales = np.abs(np.log(np.clip(keyframe_scales, 1e-6, None))) if keyframe_scales else np.empty(0)
    diagnostics: Dict[str, Any] = {
        "method": "ground_plane_lk_ransac_keyframe_chain",
        "motion_model": motion_model,
        "n_frames": n_frames,
        "keyframe_interval_frames": keyframe_interval,
        "keyframe_count": len(keyframes),
        "failed_frame_count": len(failed_frames),
        "failed_frame_frac": round(failed_frac, 4),
        "coverage": round(1.0 - failed_frac, 4),
        "keyframe_reproj_px_median": round(float(np.median(keyframe_errors)), 4) if keyframe_errors else None,
        "keyframe_reproj_px_p90": round(float(np.percentile(keyframe_errors, 90)), 4) if keyframe_errors else None,
        "keyframe_inliers_min": int(min(keyframe_inliers)) if keyframe_inliers else 0,
        "keyframe_inliers_median": float(np.median(keyframe_inliers)) if keyframe_inliers else 0.0,
        # |log similarity scale| per keyframe step: a camera that changes
        # height/zoom relative to the ground violates the rigid model
        "keyframe_scale_dev_p90": round(float(np.percentile(log_scales, 90)), 4) if len(log_scales) else None,
        # random-walk accumulation of the per-keyframe ground-plane standard error
        "drift_estimate_m": round(float(np.sqrt(np.sum(np.square(keyframe_world_errors)))), 4)
        if keyframe_world_errors else None,
        "keyframes": keyframe_log,
    }
    return transforms, diagnostics



def resolve_motion_model(camera: Mapping[str, Any], params: Optional[Mapping[str, Any]] = None) -> str:
    """Per-case motion model from the Unity camera (``unity_camera_motion``)."""
    cfg = {**EGO_DEFAULTS, **(params or {})}
    model = str(cfg.get("motion_model", "auto"))
    if model != "auto":
        return model
    rotating = float(camera.get("rotation_range_deg") or 0.0) > MOVING_CAMERA_ROT_DEG
    return "ground_rigid" if rotating else "ground_translation"


def _camera_track(transforms: Mapping[int, np.ndarray], h0: np.ndarray, centre0: np.ndarray) -> Dict[int, np.ndarray]:
    """Camera ground position per frame, ``T_t(c_0)`` with ``T_t = G_t H0^-1``."""
    h0_inv = np.linalg.inv(np.asarray(h0, dtype=np.float64))
    return {int(f): apply_homography(g @ h0_inv, centre0)[0] for f, g in transforms.items()}


def split_consistency(
    frames: Any,
    plane_from_pixel0: np.ndarray,
    transforms: Mapping[int, np.ndarray],
    centre0: np.ndarray,
    *,
    fps: float,
    object_boxes: Optional[Mapping[int, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Row-level drift diagnostic: the camera path re-estimated from the left
    and from the right image half (disjoint features) must agree.

    On the Unity references the halves agree to <= 1.6 % of the camera path
    (<= 0.6 m); where the ground texture is unstable (re-synthesised road in
    generated video) they disagree by 30-80 %, i.e. the path is not
    determined by the video and the row is flagged instead of scored.
    """
    cfg = {**EGO_DEFAULTS, **(params or {})}
    tracks = {}
    for region in ("left", "right"):
        half, _ = estimate_plane_camera_motion(
            frames, plane_from_pixel0, fps=fps, object_boxes=object_boxes, params={**cfg, "region": region}
        )
        tracks[region] = _camera_track(half, plane_from_pixel0, centre0)
    main = _camera_track(transforms, plane_from_pixel0, centre0)
    order = sorted(set(main) & set(tracks["left"]) & set(tracks["right"]))
    split = max(
        (float(np.linalg.norm(tracks["left"][f] - tracks["right"][f])) for f in order), default=float("inf")
    )
    path = float(sum(np.linalg.norm(main[b] - main[a]) for a, b in zip(order, order[1:])))
    return {
        "split_max_m": round(split, 4),
        "estimated_camera_path_m": round(path, 4),
        "split_relative": round(split / max(path, 1.0), 4),
    }


_REFERENCE_FLOOR_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def reference_camera_floor(
    reference_video: str,
    plane_from_pixel0: np.ndarray,
    poses: np.ndarray,
    *,
    object_boxes: Optional[Mapping[int, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Case-level error floor: the same estimator on the Unity reference video.

    The recovered camera path ``T_t(c_0)`` is compared with the Unity camera
    centres; ``max_error_m / path_m`` is how well the method can follow even
    Unity's own camera in this scene (low, untextured or off-plane views --
    BilliardGame's ball-level table view -- are where it cannot).  Cached per
    reference video and parameters within a process.
    """
    cfg = {**EGO_DEFAULTS, **(params or {})}
    key = (str(reference_video), json.dumps(cfg, sort_keys=True, default=str), np.asarray(plane_from_pixel0).round(9).tobytes())
    if key in _REFERENCE_FLOOR_CACHE:
        return _REFERENCE_FLOOR_CACHE[key]
    cap = cv2.VideoCapture(str(reference_video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    transforms, diag = estimate_plane_camera_motion(
        str(reference_video), plane_from_pixel0, fps=fps, object_boxes=object_boxes, params=cfg
    )
    poses = np.asarray(poses, dtype=np.float64)
    h0_inv = np.linalg.inv(np.asarray(plane_from_pixel0, dtype=np.float64))
    centre0 = poses[0][[0, 2], 3]
    errors = []
    for frame, matrix in transforms.items():
        if frame >= len(poses):
            break
        estimate = apply_homography(matrix @ h0_inv, centre0)[0]
        errors.append(float(np.linalg.norm(estimate - poses[frame][[0, 2], 3])))
    path = float(np.sum(np.linalg.norm(np.diff(poses[:, [0, 2], 3], axis=0), axis=1)))
    max_error = max(errors) if errors else float("inf")
    out = {
        "reference_video": str(reference_video),
        "camera_path_m": round(path, 4),
        "max_error_m": round(max_error, 4),
        "final_error_m": round(errors[-1], 4) if errors else None,
        "relative_max_error": round(max_error / max(path, 1.0), 4),
        "keyframe_reproj_px_p90": diag.get("keyframe_reproj_px_p90"),
    }
    _REFERENCE_FLOOR_CACHE[key] = out
    return out


def ego_reliability(
    diagnostics: Mapping[str, Any],
    params: Optional[Mapping[str, Any]] = None,
    reference_floor: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Row-level gates (catastrophic tracking failure on the candidate) plus
    the case-level reference floor (``reference_camera_floor``)."""
    cfg = {**EGO_DEFAULTS, **(params or {})}
    reasons: List[str] = []
    reproj = diagnostics.get("keyframe_reproj_px_median")
    if reproj is None or reproj > float(cfg["max_keyframe_reproj_px"]):
        reasons.append(f"keyframe_reproj_px_median={reproj} > {cfg['max_keyframe_reproj_px']}")
    inliers = float(diagnostics.get("keyframe_inliers_median") or 0.0)
    if inliers < float(cfg["min_inliers_median"]):
        reasons.append(f"keyframe_inliers_median={inliers} < {cfg['min_inliers_median']}")
    split = diagnostics.get("split_consistency")
    if isinstance(split, Mapping):
        allowed = max(
            float(cfg["max_split_m"]),
            float(cfg["max_split_relative"]) * float(split.get("estimated_camera_path_m") or 0.0),
        )
        if float(split.get("split_max_m", float("inf"))) > allowed:
            reasons.append(f"split_max_m={split.get('split_max_m')} > {round(allowed, 4)}")
    if reference_floor is not None:
        relative = reference_floor.get("relative_max_error")
        if relative is None or relative > float(cfg["max_reference_relative_error"]):
            reasons.append(
                f"reference_floor relative_max_error={relative} > {cfg['max_reference_relative_error']}"
            )
    if float(diagnostics.get("coverage") or 0.0) < float(cfg["min_coverage"]):
        reasons.append(f"coverage={diagnostics.get('coverage')} < {cfg['min_coverage']}")
    if float(diagnostics.get("failed_frame_frac") or 0.0) > float(cfg["max_failed_frame_frac"]):
        reasons.append(
            f"failed_frame_frac={diagnostics.get('failed_frame_frac')} > {cfg['max_failed_frame_frac']}"
        )
    return not reasons, reasons


def lift_points(points_by_frame: Mapping[int, np.ndarray], transforms: Mapping[int, np.ndarray]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for frame, point in points_by_frame.items():
        matrix = transforms.get(int(frame))
        if matrix is None:
            continue
        mapped = apply_homography(matrix, np.asarray(point, dtype=np.float64))[0]
        if np.all(np.isfinite(mapped)):
            out[int(frame)] = mapped
    return out


def _rotation_2x2(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64)[:2, :2])
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def anchor_contact_offset(
    lifted: Mapping[int, np.ndarray],
    reference_xz: Mapping[int, np.ndarray],
    transforms: Mapping[int, np.ndarray],
    params: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Calibrate the tracked contact point to the object origin at frame 0.

    The detector's ground-contact pixel (box bottom-centre) is not the
    object's transform origin -- for the chase-camera car it is the rear
    bumper, ~1.7 m behind the car's centre.  Frame 0 is the conditioning
    frame shared by Unity and every candidate, so the offset between the
    lifted contact point and the Unity object position there is a property of
    the conditioning view, not of the candidate's motion.  It is carried in
    the candidate camera's ground frame (rotated with the estimated camera
    yaw ``T_t``), i.e. it assumes the object is seen from a fixed relative
    direction -- exact for an object the camera follows.  Without a tracked
    point in the first ``anchor_max_frame`` frames nothing is changed.
    """
    cfg = {**EGO_DEFAULTS, **(params or {})}
    out = {int(f): np.asarray(p, dtype=np.float64) for f, p in lifted.items()}
    info: Dict[str, Any] = {"applied": False}
    if not bool(cfg.get("anchor_contact_offset", True)):
        info["reason"] = "disabled"
        return out, info
    max_frame = int(cfg.get("anchor_max_frame", 3))
    candidates = [f for f in sorted(out) if f <= max_frame and f in reference_xz and f in transforms]
    if not candidates or 0 not in transforms:
        info["reason"] = f"object not tracked in frames 0..{max_frame}"
        return out, info
    f0 = candidates[0]
    h0_inv = np.linalg.inv(transforms[0])
    offset_world = np.asarray(reference_xz[f0], dtype=np.float64) - out[f0]
    offset_camera = _rotation_2x2(transforms[f0] @ h0_inv).T @ offset_world
    for frame in list(out):
        matrix = transforms.get(frame)
        if matrix is None:
            continue
        out[frame] = out[frame] + _rotation_2x2(matrix @ h0_inv) @ offset_camera
    info.update(
        {
            "applied": True,
            "anchor_frame": int(f0),
            "offset_m": round(float(np.linalg.norm(offset_world)), 4),
        }
    )
    return out, info


def camera_centres_from_plane_homographies(
    transforms: Mapping[int, np.ndarray],
    intrinsics: np.ndarray,
    plane_y: float,
) -> Dict[int, np.ndarray]:
    """Recover the camera centre (world X/Z) implied by each G_t.

    ``G_t^-1 ~ K [r1 r3 (r2*y0 + t)]`` for the world plane y = y0 with plane
    coordinates (x, z).  Used for validation against Unity poses only.
    """
    out: Dict[int, np.ndarray] = {}
    for frame, g in transforms.items():
        idx = min(max(int(frame), 0), len(intrinsics) - 1)
        fx, fy, cx, cy = intrinsics[idx]
        k = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        try:
            m = np.linalg.inv(k) @ np.linalg.inv(g)
        except np.linalg.LinAlgError:
            continue
        lam = 2.0 / (np.linalg.norm(m[:, 0]) + np.linalg.norm(m[:, 1]))
        r1, r3, tp = lam * m[:, 0], lam * m[:, 1], lam * m[:, 2]
        if tp[2] < 0:  # plane origin must be in front of the camera
            r1, r3, tp = -r1, -r3, -tp
        r2 = np.cross(r3, r1)
        rot = np.stack([r1, r2, r3], axis=1)
        u, _, vt = np.linalg.svd(rot)
        rot = u @ vt
        t = tp - rot[:, 1] * plane_y
        centre = -rot.T @ t
        out[int(frame)] = centre
    return out


# ---------------------------------------------------------------------------
# Robust mechanics
# ---------------------------------------------------------------------------
def _odd_window(window_sec: float, fps: float, polyorder: int) -> int:
    n = int(round(window_sec * fps))
    n = max(n, polyorder + 2)
    if n % 2 == 0:
        n += 1
    return n


def _savgol_series(values: np.ndarray, window: int, polyorder: int, deriv: int, delta: float) -> np.ndarray:
    from scipy.signal import savgol_filter

    n = len(values)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= polyorder:
        if deriv == 0:
            return values.copy()
        return np.gradient(values, delta, axis=0)
    return savgol_filter(values, w, polyorder, deriv=deriv, delta=delta, axis=0, mode="interp")


def smooth_kinematics(
    points_by_frame: Mapping[int, np.ndarray],
    fps: float,
    *,
    window_sec: float,
    polyorder: int = 2,
    max_gap_frames: int = 3,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Savitzky--Golay position/velocity per frame over contiguous runs.

    Gaps up to ``max_gap_frames`` are linearly interpolated; longer gaps split
    the trajectory into separately smoothed runs (no velocity across a gap).
    """
    frames = sorted(int(f) for f in points_by_frame)
    if len(frames) < 2 or fps <= 0:
        return {}
    runs: List[List[int]] = [[frames[0]]]
    for f in frames[1:]:
        if f - runs[-1][-1] <= max_gap_frames + 1:
            runs[-1].append(f)
        else:
            runs.append([f])
    window = _odd_window(window_sec, fps, polyorder)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for run in runs:
        if len(run) < 3:
            continue
        dense = np.arange(run[0], run[-1] + 1)
        known = np.asarray(run)
        values = np.stack([np.asarray(points_by_frame[f], dtype=np.float64) for f in run])
        filled = np.stack(
            [np.interp(dense, known, values[:, d]) for d in range(values.shape[1])], axis=1
        )
        position = _savgol_series(filled, window, polyorder, 0, 1.0 / fps)
        velocity = _savgol_series(filled, window, polyorder, 1, 1.0 / fps)
        observed = set(run)
        for i, f in enumerate(dense):
            if int(f) in observed:
                out[int(f)] = {"position": position[i], "velocity": velocity[i]}
    return out


def _linear_fit_velocity(frames: Sequence[int], points: np.ndarray, fps: float) -> Optional[np.ndarray]:
    if len(frames) < 3:
        return None
    t = np.asarray(frames, dtype=np.float64) / fps
    tc = t - t.mean()
    denom = float(np.sum(tc ** 2))
    if denom <= EPS:
        return None
    return (tc @ (points - points.mean(axis=0))) / denom


def _quadratic_fit_accel(frames: Sequence[int], points: np.ndarray, fps: float) -> float:
    """|acceleration| of a least-squares quadratic over a short window."""
    if len(frames) < 4:
        return 0.0
    t = np.asarray(frames, dtype=np.float64) / fps
    t = t - t.mean()
    design = np.stack([np.ones_like(t), t, t ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(design, points, rcond=None)
    return float(np.linalg.norm(2.0 * coef[2]))


def detect_contact_events(
    gt_points: Mapping[int, np.ndarray],
    fps: float,
    *,
    peak_speed: float,
    params: Optional[Mapping[str, Any]] = None,
) -> List[int]:
    """GT frames with an impulsive velocity change (collisions / strikes).

    A contact is a local maximum of the raw GT one-frame acceleration that
    exceeds ``contact_accel_frac * peak_speed`` per second and changes the
    velocity by at least ``contact_min_dv_frac * peak_speed`` between linear
    fits before and after it.  Unity transforms are noise-free, so raw
    differences are safe on the GT side.
    """
    cfg = {**ROBUST_DEFAULTS, **(params or {})}
    frames = sorted(int(f) for f in gt_points)
    if len(frames) < 5 or peak_speed <= 0:
        return []
    pos = np.stack([np.asarray(gt_points[f], dtype=np.float64) for f in frames])
    vel = {}
    for (f0, p0), (f1, p1) in zip(zip(frames, pos), zip(frames[1:], pos[1:])):
        if f1 - f0 == 1:
            vel[f1] = (p1 - p0) * fps
    accel = {}
    for f in vel:
        if f + 1 in vel:
            accel[f] = float(np.linalg.norm(vel[f + 1] - vel[f]) * fps)
    threshold = float(cfg["contact_accel_frac"]) * peak_speed
    window = max(3, int(round(float(cfg["contact_fit_window_sec"]) * fps)))
    guard = max(1, int(round(float(cfg["contact_guard_sec"]) * fps)))
    events: List[int] = []
    for f in sorted(accel):
        a = accel[f]
        if a < threshold:
            continue
        if any(accel.get(g, 0.0) > a for g in range(f - guard, f + guard + 1) if g != f):
            continue
        if events and f - events[-1] <= 2 * guard:
            continue
        before = [g for g in range(f - guard - window, f - guard + 1) if g in gt_points]
        after = [g for g in range(f + guard, f + guard + window + 1) if g in gt_points]
        vb = _linear_fit_velocity(before, np.stack([gt_points[g] for g in before]), fps) if len(before) >= 3 else None
        va = _linear_fit_velocity(after, np.stack([gt_points[g] for g in after]), fps) if len(after) >= 3 else None
        if vb is None or va is None:
            continue
        dv_norm = float(np.linalg.norm(va - vb))
        if dv_norm < float(cfg["contact_min_dv_frac"]) * peak_speed:
            continue
        # Impulsive, not a smooth acceleration (braking, throttle): the jump
        # across the contact must exceed what the acceleration *within* the
        # before/after windows would accumulate over the same gap.
        gap = (np.mean(after) - np.mean(before)) / fps
        ab = _quadratic_fit_accel(before, np.stack([gt_points[g] for g in before]), fps)
        aa = _quadratic_fit_accel(after, np.stack([gt_points[g] for g in after]), fps)
        smooth = max(ab, aa) * gap
        if dv_norm < float(cfg["contact_impulsive_ratio"]) * smooth:
            continue
        events.append(f)
    return events


def robust_mechanics(
    candidate_points: Mapping[int, np.ndarray],
    gt_points: Mapping[int, np.ndarray],
    fps: float,
    *,
    mass_kg: float,
    alignment_tolerance_frames: int = 5,
    params: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Span-averaged, peak-normalised mechanics from smoothed trajectories."""
    cfg = {**ROBUST_DEFAULTS, **(params or {})}
    window_sec = float(cfg["smoothing_window_sec"])
    polyorder = int(cfg["polyorder"])
    cand = smooth_kinematics(candidate_points, fps, window_sec=window_sec, polyorder=polyorder)
    gt = smooth_kinematics(gt_points, fps, window_sec=window_sec, polyorder=polyorder)
    if not cand or not gt:
        return None
    gt_speed = {f: float(np.linalg.norm(v["velocity"])) for f, v in gt.items()}
    cand_speed = {f: float(np.linalg.norm(v["velocity"])) for f, v in cand.items()}
    peak_frame = max(gt_speed, key=lambda f: (gt_speed[f], -f))
    peak_speed = gt_speed[peak_frame]
    if peak_speed < float(cfg["moving_speed_floor_m_s"]):
        return {"status": "object_never_moves", "peak_speed_m_s": round(peak_speed, 6)}
    threshold = max(float(cfg["moving_speed_frac"]) * peak_speed, float(cfg["moving_speed_floor_m_s"]))
    # Bounded motion-onset offset (world-model action latency): the first
    # frame the candidate's smoothed speed reaches the moving threshold vs the
    # GT's, clamped to the module's event-alignment tolerance.  Onsets are
    # sharp (a strike, a launch); a broad speed peak is not, and aligning on
    # it shifted even a self-match by several frames.
    tol = max(0, int(alignment_tolerance_frames))
    gt_onset = next((f for f in sorted(gt_speed) if gt_speed[f] >= threshold), None)
    # a real transition only: the frame before must be observed and below
    # the threshold (re-acquiring a fast object after a detector gap is not
    # an onset)
    cand_onset = next(
        (
            f
            for f in sorted(cand_speed)
            if gt_onset is not None
            and abs(f - gt_onset) <= tol
            and cand_speed[f] >= threshold
            and cand_speed.get(f - 1, float("inf")) < threshold
        ),
        None,
    )
    offset = int(cand_onset - gt_onset) if cand_onset is not None and gt_onset is not None else 0
    span = [f for f in sorted(gt_speed) if gt_speed[f] >= threshold]
    # Like with like: the GT is resampled on the candidate's observed frames
    # (detector gaps at a strike or behind an occluder) and smoothed with the
    # same filter, so a self-match differs only by tracking/lifting noise.
    gt_on_support = {
        int(f): gt_points[f] for f in gt_points if (int(f) + offset) in candidate_points
    }
    gt_matched = smooth_kinematics(gt_on_support, fps, window_sec=window_sec, polyorder=polyorder)
    gt_matched_speed = {f: float(np.linalg.norm(v["velocity"])) for f, v in gt_matched.items()}
    pairs = [
        (f, f + offset) for f in span if f in gt_matched_speed and (f + offset) in cand_speed
    ]
    coverage = len(pairs) / max(len(span), 1)
    if len(pairs) < 3 or coverage < float(cfg["min_span_coverage"]):
        return {
            "status": "insufficient_overlap",
            "moving_span_frames": len(span),
            "paired_frames": len(pairs),
            "span_coverage": round(coverage, 4),
            "peak_speed_gt_m_s": round(peak_speed, 6),
        }
    v_gt = np.asarray([gt_matched_speed[f] for f, _ in pairs])
    v_hat = np.asarray([cand_speed[g] for _, g in pairs])
    momentum_err = float(np.mean(np.abs(v_hat - v_gt)) / peak_speed)
    energy_err = float(np.mean(np.abs(v_hat ** 2 - v_gt ** 2)) / peak_speed ** 2)
    gt_span_peak = float(np.max(v_gt))
    cand_span_peak = float(np.max(v_hat))
    peak_speed_err = abs(cand_span_peak - gt_span_peak) / max(gt_span_peak, EPS)

    # contact events: momentum change across each GT contact, linear fits
    events = detect_contact_events(gt_points, fps, peak_speed=peak_speed, params=cfg)
    fit_window = max(3, int(round(float(cfg["contact_fit_window_sec"]) * fps)))
    guard = max(1, int(round(float(cfg["contact_guard_sec"]) * fps)))
    contact_rows: List[Dict[str, Any]] = []
    for event in events:
        def dv(points: Mapping[int, np.ndarray], centre: int) -> Optional[np.ndarray]:
            before = [g for g in range(centre - guard - fit_window, centre - guard + 1) if g in points]
            after = [g for g in range(centre + guard, centre + guard + fit_window + 1) if g in points]
            if len(before) < 3 or len(after) < 3:
                return None
            vb = _linear_fit_velocity(before, np.stack([np.asarray(points[g], dtype=np.float64) for g in before]), fps)
            va = _linear_fit_velocity(after, np.stack([np.asarray(points[g], dtype=np.float64) for g in after]), fps)
            if vb is None or va is None:
                return None
            return va - vb
        dv_gt = dv(gt_points, event)
        dv_hat = dv(candidate_points, event + offset)
        if dv_gt is None:
            continue
        row: Dict[str, Any] = {
            "frame": int(event),
            "dp_gt_kg_m_s": round(float(mass_kg * np.linalg.norm(dv_gt)), 6),
        }
        if dv_hat is not None:
            row["dp_hat_kg_m_s"] = round(float(mass_kg * np.linalg.norm(dv_hat)), 6)
            row["dp_error_pct"] = round(
                100.0 * float(np.linalg.norm(dv_hat - dv_gt) / max(np.linalg.norm(dv_gt), EPS)), 4
            )
        contact_rows.append(row)
    contact_errors = [r["dp_error_pct"] for r in contact_rows if "dp_error_pct" in r]
    return {
        "status": "computed",
        "estimator": "savgol_span_averaged_peak_normalised",
        "smoothing_window_frames": _odd_window(window_sec, fps, polyorder),
        "smoothing_window_sec": window_sec,
        "moving_speed_threshold_m_s": round(threshold, 6),
        "moving_span_frames": len(span),
        "paired_frames": len(pairs),
        "span_coverage": round(coverage, 4),
        "temporal_offset_frames": offset,
        "peak_speed_gt_m_s": round(peak_speed, 6),
        "peak_speed_gt_observed_m_s": round(gt_span_peak, 6),
        "peak_speed_hat_m_s": round(cand_span_peak, 6),
        "peak_momentum_gt_kg_m_s": round(mass_kg * peak_speed, 6),
        "peak_energy_gt_j": round(0.5 * mass_kg * peak_speed ** 2, 6),
        "robust_momentum_error_pct": round(100.0 * momentum_err, 4),
        "robust_energy_error_pct": round(100.0 * energy_err, 4),
        "robust_error_rate_pct": round(50.0 * (momentum_err + energy_err), 4),
        "peak_speed_error_pct": round(100.0 * peak_speed_err, 4),
        "contact_events": contact_rows,
        "contact_dp_error_pct": round(float(np.mean(contact_errors)), 4) if contact_errors else None,
    }
