"""Analytic camera-to-world poses for a fixed-target circular orbit.

Right-handed world, +Y up; OpenCV camera +X right, +Y down, +Z forward.
Positive phase follows the right-hand rule about +Y (CCW from above).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def validate_orbit(raw: Mapping[str, Any]) -> None:
    if raw.get("target") != "camera":
        raise ValueError("camera_look_at_orbit requires target: camera")
    if raw.get("coordinates") != "right_handed_y_up_opencv_c2w":
        raise ValueError("orbit coordinates must be right_handed_y_up_opencv_c2w")
    if set(raw["channels"]) != {"orbit_phase_rate_deg_s"}:
        raise ValueError("orbit requires only orbit_phase_rate_deg_s; yaw is not orbital phase")
    geometry = raw.get("geometry", {})
    for key in ("origin_m", "look_at_m"):
        a = np.asarray(geometry.get(key), dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError(f"geometry.{key} must be a finite 3-vector")
    for key in ("radius_m", "eye_height_m", "initial_phase_deg", "vertical_fov_deg"):
        value = geometry.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError(f"geometry.{key} must be finite numeric")
    if geometry["radius_m"] <= 0 or not 0 < geometry["vertical_fov_deg"] < 180:
        raise ValueError("invalid orbit radius/FOV")
    origin, target = np.asarray(geometry["origin_m"]), np.asarray(geometry["look_at_m"])
    if not np.allclose(origin[[0, 2]], target[[0, 2]], atol=1e-10, rtol=0):
        raise ValueError("look-at target must lie on the vertical orbit axis")


def orbit_poses(raw: Mapping[str, Any], times_sec: np.ndarray) -> np.ndarray:
    """Evaluate at native timestamps directly, including an optional endpoint."""
    validate_orbit(raw)
    times = np.asarray(times_sec, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0) or np.any(times > float(raw["duration_sec"])):
        raise ValueError("pose timestamps must be finite and within the action duration")
    g = raw["geometry"]
    phase = np.full(times.shape, float(g["initial_phase_deg"]))
    for segment in raw["segments"]:
        dt = np.clip(times - float(segment["start_sec"]), 0,
                     float(segment["end_sec"]) - float(segment["start_sec"]))
        phase += dt * float(segment["values"]["orbit_phase_rate_deg_s"])
    radians = np.deg2rad(phase)
    positions = np.asarray(g["origin_m"], dtype=float) + np.column_stack((
        float(g["radius_m"]) * np.sin(radians),
        np.full(len(times), float(g["eye_height_m"])),
        float(g["radius_m"]) * np.cos(radians)))
    forward = np.asarray(g["look_at_m"], dtype=float) - positions
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    down = np.cross(forward, right)
    result = np.tile(np.eye(4), (len(times), 1, 1))
    result[:, :3, :3] = np.stack((right, down, forward), axis=-1)
    result[:, :3, 3] = positions
    return result


def camera_intrinsics(raw: Mapping[str, Any], width: int, height: int) -> np.ndarray:
    """Square pixels with a vertical FOV; resize/crop adapters must transform K."""
    validate_orbit(raw)
    if min(width, height) <= 0:
        raise ValueError("image dimensions must be positive")
    focal = height / (2 * np.tan(np.deg2rad(raw["geometry"]["vertical_fov_deg"]) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
