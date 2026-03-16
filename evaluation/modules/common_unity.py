from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_unity_log(log_path: str) -> List[Dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Unity log does not exist: {log_path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Unity log should be a list of frames, got: {type(payload)}")
    return payload


def _rx(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _ry(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rz(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def unity_euler_deg_to_light_direction(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """
    Approximate Unity inspector Euler angles (x, y, z) as Y-X-Z composition.
    Directional light in Unity points along local -Z.
    """
    x = math.radians(x_deg)
    y = math.radians(y_deg)
    z = math.radians(z_deg)
    rot = _ry(y) @ _rx(x) @ _rz(z)
    forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    light_dir = -(rot @ forward)
    norm = np.linalg.norm(light_dir)
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    return light_dir / norm


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return float("nan")
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def sample_series_by_count(series: Sequence[Any], count: int) -> List[Any]:
    if count <= 0 or len(series) == 0:
        return []
    if len(series) == count:
        return list(series)
    idx = np.linspace(0, len(series) - 1, num=count)
    return [series[int(round(i))] for i in idx]


def extract_light_vectors_and_intensity(log_frames: Sequence[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[float]]:
    vectors: List[np.ndarray] = []
    intensity: List[float] = []
    for frame in log_frames:
        e = frame.get("lightEulerAngles") or {}
        x = float(e.get("x", 0.0))
        y = float(e.get("y", 0.0))
        z = float(e.get("z", 0.0))
        vectors.append(unity_euler_deg_to_light_direction(x, y, z))
        intensity.append(float(frame.get("lightIntensity", 0.0)))
    return vectors, intensity


def get_unity_log_path(artifacts: Dict[str, Any]) -> Optional[str]:
    value = artifacts.get("unity_log_json")
    if value is None:
        return None
    value = str(value).strip()
    return value or None
