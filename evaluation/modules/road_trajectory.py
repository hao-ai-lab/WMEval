"""Ego-vehicle trajectory in road coordinates for chase-camera driving scenes.

A chase camera keeps the car near the image centre, so the car's own pixel track
says almost nothing about where it went. The road says it instead. On a straight
road the vehicle's path is two numbers per frame: distance along the road and
position across it. Both are measured from the video alone:

1. Each frame is projected to a top-down view of the ground with the camera rig
   of the conditioning frame (intrinsics, height and pitch from Unity frame 0;
   a chase camera keeps them fixed relative to the car).
2. Road direction. Lane markings are parallel to the road, so the heading of the
   camera relative to the road is the rotation that makes the marking pixels'
   lateral histogram sharpest. This is absolute in every frame: no drift.
3. Rig motion. Consecutive top-down views, rotated to road axes, are aligned by a
   masked translation (ECC); the shift is the camera's motion along and across
   the road.
4. The car is located from its body colour and lifted through the same ground
   projection. Where it is not visible it keeps its last observed position (the
   camera has driven past it or it left the view), counted in ``car_visible_frac``.

Nothing after frame 0 uses Unity's camera, so a video whose car drives the wrong
way gets a wrong trajectory (the legacy metric re-projected the candidate track
through Unity's own camera and reproduced Unity's path for any centred car).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

BEV_RES_M = 0.05
BEV_X = (-12.0, 12.0)           # lateral extent relative to the camera, metres
BEV_Z = (5.0, 24.0)             # forward extent relative to the camera, metres
HEADING_SEARCH_DEG = 12.0       # per-frame search window around the previous heading
HEADING_STEP_DEG = 0.5
LATERAL_BIN_M = 0.1
MIN_MARKING_PIXELS = 400
MIN_ECC_CC = 0.5
NOMINAL_CAR_OFFSET_M = np.array([0.0, 8.0])
DUPLICATE_TOL = 0.25            # mean abs grey difference below which a frame repeats the last one
MAX_JUMP_PX = 12                # largest change in per-frame shift vs the recent median (0.6 m)


@dataclass(frozen=True)
class Rig:
    """Camera intrinsics and its frame-0 orientation over a flat ground plane.

    ``R`` holds the camera axes (x right, y down, z forward) as columns in world
    axes with y up, as Unity logs them (a left-handed basis, used as given)."""

    K: np.ndarray
    R: np.ndarray
    height_m: float
    image_size: Tuple[int, int]

    @classmethod
    def from_unity(cls, intrinsics: np.ndarray, pose0: np.ndarray, plane_y: float,
                   image_size: Tuple[int, int]) -> "Rig":
        fx, fy, cx, cy = [float(v) for v in np.asarray(intrinsics).reshape(-1, 4)[0]]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return cls(K=K, R=np.asarray(pose0[:3, :3], dtype=np.float64),
                   height_m=float(pose0[1, 3] - plane_y), image_size=image_size)

    @property
    def pitch_deg(self) -> float:
        return float(np.degrees(np.arcsin(-self.R[1, 2])))

    def with_pitch_height(self, pitch_deg: float, height_m: float) -> "Rig":
        t = np.radians(pitch_deg)
        R = np.array([[1, 0, 0], [0, -np.cos(t), -np.sin(t)], [0, -np.sin(t), np.cos(t)]], dtype=np.float64)
        return Rig(K=self.K, R=R, height_m=height_m, image_size=self.image_size)

    def image_from_ground(self) -> np.ndarray:
        """Homography from rig ground coords (X right, Z forward from the camera, metres) to pixels."""
        Rt = self.R.T
        return self.K @ np.stack([Rt[:, 0], Rt[:, 2], -self.height_m * Rt[:, 1]], axis=1)

    def ground_from_image(self) -> np.ndarray:
        return np.linalg.inv(self.image_from_ground())


def bev_homography(rig: Rig) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Homography from top-down pixel (col, row) to image pixel, and the top-down size."""
    w = int(round((BEV_X[1] - BEV_X[0]) / BEV_RES_M))
    h = int(round((BEV_Z[1] - BEV_Z[0]) / BEV_RES_M))
    ground_from_bev = np.array([[BEV_RES_M, 0, BEV_X[0]], [0, -BEV_RES_M, BEV_Z[1]], [0, 0, 1]], dtype=np.float64)
    return rig.image_from_ground() @ ground_from_bev, (w, h)


def bev_to_ground(cols: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return BEV_X[0] + cols * BEV_RES_M, BEV_Z[1] - rows * BEV_RES_M


def off_road_mask(bgr: np.ndarray) -> np.ndarray:
    """Saturated pixels: not asphalt or markings (car, hovering cube, grass, sky)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 70, 0), (180, 255, 255))


def car_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (15, 110, 110), (38, 255, 255))


def marking_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 175), (180, 55, 255))


def _car_box(mask: np.ndarray, min_px: int) -> Optional[Tuple[int, int, int, int]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_px:
        return None
    x, y, w, h = stats[i, :4]
    return int(x), int(y), int(w), int(h)


def _heading(xs: np.ndarray, zs: np.ndarray, centre_deg: float, half_deg: float) -> Tuple[float, float]:
    """Road heading (deg, camera relative) maximising lateral-histogram sharpness."""
    best = (centre_deg, -1.0)
    for deg in np.arange(centre_deg - half_deg, centre_deg + half_deg + 1e-9, HEADING_STEP_DEG):
        t = np.radians(deg)
        lat = xs * np.cos(t) - zs * np.sin(t)
        hist = np.bincount(np.floor((lat - lat.min()) / LATERAL_BIN_M).astype(int))
        score = float((hist.astype(np.float64) ** 2).sum()) / max(len(xs), 1) ** 2
        if score > best[1]:
            best = (float(deg), score)
    return best


def _road_aligned(bev: np.ndarray, valid: np.ndarray, heading_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a top-down view about the camera so the road runs straight up the image."""
    h, w = bev.shape[:2]
    centre = (-BEV_X[0] / BEV_RES_M, BEV_Z[1] / BEV_RES_M)        # camera position in top-down pixels
    M = cv2.getRotationMatrix2D(centre, heading_deg, 1.0)      # markings tilt clockwise by the heading
    return (cv2.warpAffine(bev, M, (w, h), flags=cv2.INTER_LINEAR),
            cv2.warpAffine(valid, M, (w, h), flags=cv2.INTER_NEAREST))


def _ecc_shift(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray, guess: Tuple[float, float]) -> Optional[Tuple[float, float, float]]:
    warp = np.array([[1, 0, guess[0]], [0, 1, guess[1]]], dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(prev, cur, warp, cv2.MOTION_TRANSLATION,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5), mask, 5)
    except cv2.error:
        return None
    return float(warp[0, 2]), float(warp[1, 2]), float(cc)


def _horizon_row(frames: Sequence[np.ndarray]) -> Optional[float]:
    """Image row where the lane markings converge (the road's vanishing point)."""
    lines = []
    for bgr in frames:
        m = marking_mask(bgr)
        h = m.shape[0]
        m[: int(h * 0.3)] = 0
        m &= ~cv2.dilate(car_mask(bgr) | off_road_mask(bgr), np.ones((15, 15), np.uint8))
        segs = cv2.HoughLinesP(m, 1, np.pi / 360, 40, minLineLength=40, maxLineGap=25)
        for x1, y1, x2, y2 in ([] if segs is None else segs.reshape(-1, 4).astype(float)):
            ang = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if 12 < ang < 85:                                   # road-parallel markings, not dash ends
                lines.append(np.cross([x1, y1, 1.0], [x2, y2, 1.0]))
    if len(lines) < 4:
        return None
    L = np.array(lines)
    L /= np.linalg.norm(L[:, :2], axis=1, keepdims=True)
    rows = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            p = np.cross(L[i], L[j])
            if abs(p[2]) < 1e-9:
                continue
            u, v = p[0] / p[2], p[1] / p[2]
            if -1000 < u < 3000 and -500 < v < frames[0].shape[0]:
                rows.append(v)
    return float(np.median(rows)) if rows else None


def lane_period(xs: np.ndarray, zs: np.ndarray, heading_deg: float, merge: float = 0.3) -> Optional[float]:
    """Median spacing of adjacent lane lines in ground units (double lines merged)."""
    t = np.radians(heading_deg)
    lat = xs * np.cos(t) - zs * np.sin(t)
    bin_w = 0.02
    lo = lat.min()
    hist = np.convolve(np.bincount(np.floor((lat - lo) / bin_w).astype(int)), np.ones(5), "same")
    peaks = [i for i in range(1, len(hist) - 1) if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]
             and hist[i] > 0.25 * hist.max()]
    pos: List[float] = []
    for i in peaks:
        x = lo + (i + 0.5) * bin_w
        if pos and x - pos[-1][-1] < merge:
            pos[-1].append(x)
        else:
            pos.append([x])
    centres = [float(np.mean(p)) for p in pos]
    if len(centres) < 2:
        return None
    return float(np.median(np.diff(centres)))


def calibrate(frames: Sequence[np.ndarray], base: Rig, lane_spacing_m: float) -> Tuple[Rig, Dict[str, Any]]:
    """Fit the camera's pitch and height to a video's own first frames.

    A generated video need not keep the conditioning frame's camera. With the
    intrinsics fixed, the pitch follows from the row where the lane markings
    converge, and the height scales the lane spacing to the scene's known value."""
    v_h = _horizon_row(frames)
    if v_h is None:
        return base, {"status": "no_markings"}
    fy, cy = base.K[1, 1], base.K[1, 2]
    pitch = float(np.degrees(np.arctan((cy - v_h) / fy)))
    rig = base.with_pitch_height(pitch, 1.0)
    periods, headings = [], []
    for bgr in frames:                       # one estimate per frame; the median resists aliasing
        m = marking_mask(bgr)
        m[: int(base.image_size[1] * 0.5)] = 0
        m &= ~cv2.dilate(car_mask(bgr) | off_road_mask(bgr), np.ones((15, 15), np.uint8))
        v, u = np.nonzero(m)
        g = rig.ground_from_image() @ np.stack([u, v, np.ones_like(u)], 0).astype(np.float64)
        ok = g[2] > 0
        x, z = g[0, ok] / g[2, ok], g[1, ok] / g[2, ok]
        keep = (z > 0) & (z < 8) & (np.abs(x) < 4)
        if keep.sum() < MIN_MARKING_PIXELS:
            continue
        heading, _ = _heading(x[keep], z[keep], 0.0, 5.0)
        p = lane_period(x[keep], z[keep], heading)
        if p is not None:
            periods.append(p)
            headings.append(heading)
    if not periods:
        return base, {"status": "no_lane_period"}
    period, heading = float(np.median(periods)), float(np.median(headings))
    height = lane_spacing_m / period
    return base.with_pitch_height(pitch, height), {
        "status": "ok", "horizon_row": round(v_h, 1), "pitch_deg": round(pitch, 2), "height_m": round(height, 3),
        "frame0_heading_deg": round(heading, 2)}


def distinct_frames(frames: Sequence[np.ndarray], tol: float = DUPLICATE_TOL) -> List[int]:
    """Indices of frames that differ from the last kept one (drops frame-rate-upsampling repeats)."""
    keep = [0]
    last = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for i in range(1, len(frames)):
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if float(np.abs(g - last).mean()) > tol:
            keep.append(i)
            last = g
    return keep


def measure(frames: Sequence[np.ndarray], rig: Rig, min_car_px: int = 1500) -> Dict[str, Any]:
    """Per-frame camera and car positions in road coordinates (x across, z along; metres).

    ``frames`` are BGR images already in the conditioning frame's pixel geometry.
    Repeated frames carry no motion, so the road is aligned between distinct frames
    only and positions are interpolated back onto every frame."""
    keep = distinct_frames(frames)
    res = _measure_distinct([frames[i] for i in keep], rig, min_car_px, gaps=[1] + list(np.diff(keep)))
    n = len(frames)
    idx = np.arange(n)
    if len(keep) == 1:
        interp = lambda a: np.repeat(a[:1], n, axis=0)
    else:
        interp = lambda a: (np.stack([np.interp(idx, keep, a[:, k]) for k in range(a.shape[1])], 1)
                            if a.ndim == 2 else np.interp(idx, keep, a))
    nearest = np.searchsorted(keep, idx, side="right") - 1
    res.update({
        "camera_xz": interp(res["camera_xz"]), "car_xz": interp(res["car_xz"]), "yaw_deg": interp(res["yaw_deg"]),
        "car_visible": res["car_visible"][nearest], "car_visible_frac": float(np.mean(res["car_visible"][nearest])),
        "distinct_frames": len(keep),
    })
    return res


def _phase_shift(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    win = cv2.createHanningWindow(prev.shape[::-1], cv2.CV_32F)
    m = (mask > 0).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(prev * m, cur * m, win)
    return float(dx), float(dy)


def _measure_distinct(frames: Sequence[np.ndarray], rig: Rig, min_car_px: int,
                      gaps: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    H, size = bev_homography(rig)
    ground_from_image = rig.ground_from_image()
    heading = 0.0
    cam = np.zeros(2)
    prev = None
    history: List[Tuple[float, float]] = []
    out_cam, out_car, out_heading, out_cc, visible = [], [], [], [], []
    car_world = None
    frame_h = rig.image_size[1]
    failed = 0
    gaps = list(gaps) if gaps is not None else [1] * len(frames)
    for k, bgr in enumerate(frames):
        gap = max(int(gaps[k]), 1)
        cmask = car_mask(bgr)
        box = _car_box(cmask, min_car_px)
        exclude = np.zeros(cmask.shape, np.uint8)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(exclude, (x - w // 3, y - h // 3), (x + w + w // 3, y + h + h // 2), 255, -1)
        exclude |= cv2.dilate(cmask, np.ones((15, 15), np.uint8))
        exclude |= cv2.dilate(off_road_mask(bgr), np.ones((9, 9), np.uint8))   # objects above the road, grass
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        warp = dict(dsize=size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        bev = cv2.warpPerspective(gray, H, **warp).astype(np.float32)
        valid = cv2.warpPerspective(np.full_like(gray, 255), H, borderValue=0, **warp)
        valid &= ~cv2.warpPerspective(exclude, H, **warp)
        marks = cv2.warpPerspective(marking_mask(bgr), H, **warp) & valid
        rows, cols = np.nonzero(marks > 0)
        if len(rows) >= MIN_MARKING_PIXELS:
            xs, zs = bev_to_ground(cols.astype(np.float64), rows.astype(np.float64))
            heading, _ = _heading(xs, zs, heading, HEADING_SEARCH_DEG if prev is not None else 30.0)
        aligned, avalid = _road_aligned(cv2.GaussianBlur(bev, (0, 0), 1.5), valid, heading)
        cc = None
        if prev is not None:
            mask = cv2.erode(avalid & prev[1], np.ones((9, 9), np.uint8))
            per_frame = np.median(np.array(history[-6:]), axis=0) if history else np.zeros(2)
            pred = tuple(per_frame * gap)
            tol = MAX_JUMP_PX * (1.0 + 0.5 * (gap - 1))
            best = None
            for guess in (pred, (0.0, 0.0), "phase"):
                if guess == "phase":
                    if best is not None:
                        break
                    guess = _phase_shift(prev[0], aligned, mask)
                res = _ecc_shift(prev[0], aligned, mask, guess)
                if res is None or res[2] < MIN_ECC_CC or max(abs(res[0] - pred[0]), abs(res[1] - pred[1])) > tol:
                    continue
                if best is None or res[2] > best[2]:
                    best = res
            if best is not None:
                dx_px, dy_px, cc = best
                history.append((dx_px / gap, dy_px / gap))
            else:                                   # no plausible alignment: carry the predicted motion
                dx_px, dy_px = pred
                failed += 1
            # cur(x + d) = prev(x): the road moved by d. Rows grow towards the camera, so
            # driving forward moves the road down (+d_row) and driving right moves it left (-d_col).
            cam = cam + np.array([-dx_px * BEV_RES_M, dy_px * BEV_RES_M])
        prev = (aligned, avalid)
        yaw = np.radians(-heading)                  # camera yaw relative to the road, positive = turned right
        # rig axes in road coords: right = (cos, -sin), forward = (sin, cos)
        R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        if box is not None and box[1] + box[3] < frame_h - 2:          # a box cut by the bottom edge has no ground contact
            x, y, w, h = box
            g = ground_from_image @ np.array([x + w / 2.0, y + h, 1.0])
            car_world = cam + R @ np.array([g[0] / g[2], g[1] / g[2]])
            visible.append(True)
        else:
            # Not visible: hold the last observed position (a chase camera that loses its car
            # has usually driven past it); before the first sighting, the nominal rig offset.
            car_world = car_world if car_world is not None else cam + R @ NOMINAL_CAR_OFFSET_M
            visible.append(False)
        out_cam.append(cam.copy())
        out_car.append(car_world.copy())
        out_heading.append(-heading)
        out_cc.append(cc)
    ccs = [c for c in out_cc if c is not None]
    return {
        "camera_xz": np.array(out_cam),
        "car_xz": np.array(out_car),
        "yaw_deg": np.array(out_heading),
        "align_failed_frac": failed / max(len(frames) - 1, 1),
        "car_visible": np.array(visible),
        "car_visible_frac": float(np.mean(visible)) if visible else 0.0,
        "ecc_cc_median": float(np.median(ccs)) if ccs else None,
        "ecc_cc_p10": float(np.percentile(ccs, 10)) if ccs else None,
    }


# Alignment failures on the Unity reference are 1-5 % (30 and 15 fps); generated roads whose
# texture is redrawn between frames fail far more often, and the path is then fill-in.
MAX_ALIGN_FAILED_FRAC = 0.25


def score_video(video_path: str, transforms_json: str, poses_npy: str, intrinsics_npy: str,
                plane_calibration_json: str, label: str = "car") -> Dict[str, Any]:
    """Final displacement of ``label`` in road coordinates against Unity's logged path.

    Both paths start at the origin and the candidate path is resampled onto the reference
    timeline by fraction of the clip. ``fde_road_m`` is withheld (None, status
    ``unreliable_alignment``) when more than MAX_ALIGN_FAILED_FRAC of the road alignments fail."""
    import json
    objects = json.loads(open(transforms_json, encoding="utf-8").read())["objects"]
    if label not in objects or "positions_world" not in objects[label]:
        return {"status": "not_applicable", "fde_road_m": None}
    pos = np.asarray(objects[label]["positions_world"], dtype=np.float64)
    gt = pos[:, [0, 2]] - pos[0, [0, 2]]
    calib = json.loads(open(plane_calibration_json, encoding="utf-8").read())
    width, height = calib.get("image_size", [1280, 720])
    rig = Rig.from_unity(np.load(intrinsics_npy), np.load(poses_npy)[0], float(calib["plane"]["value_m"]),
                         (int(width), int(height)))
    cap, frames = cv2.VideoCapture(video_path), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (int(width), int(height))) if frame.shape[1::-1] != (width, height) else frame)
    if len(frames) < 2:
        return {"status": "no_frames", "fde_road_m": None}
    res = measure(frames, rig)
    hat = res["car_xz"] - res["car_xz"][0]
    grid = np.linspace(0.0, 1.0, len(gt))
    src = np.linspace(0.0, 1.0, len(hat))
    hat = np.stack([np.interp(grid, src, hat[:, k]) for k in range(2)], axis=1)
    reliable = res["align_failed_frac"] <= MAX_ALIGN_FAILED_FRAC
    fde = float(np.linalg.norm(hat[-1] - gt[-1]))
    return {"status": "computed" if reliable else "unreliable_alignment",
            "fde_road_m": round(fde, 6) if reliable else None, "fde_road_m_unchecked": round(fde, 6),
            "align_failed_frac": round(float(res["align_failed_frac"]), 4),
            "ecc_cc_median": round(float(res["ecc_cc_median"]), 4),
            "car_visible_frac": round(float(res["car_visible_frac"]), 4),
            "end_xz": [round(float(v), 3) for v in hat[-1]], "reference_end_xz": [round(float(v), 3) for v in gt[-1]]}
