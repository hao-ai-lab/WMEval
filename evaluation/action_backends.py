"""Backend registry for compiling canonical action trajectories.

The semantic trajectory YAMLs (``experiments/tables_4_7/trajectories``) say
*what* happens over time.  ``experiments/tables_4_7/action_backends.yaml`` says
*which entity* each trajectory actuates and *how* each backend can express it:
its native timeline, supported action targets, and frozen calibration.  Input
preparation for every model goes through this module so that capability checks
and calibration constants live in one reviewed place.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "experiments/tables_4_7/action_backends.yaml"
TARGETS = ("camera", "ego_vehicle", "cue_ball", "table_objects")


class UnsupportedAction(ValueError):
    """Raised when a backend cannot express a trajectory's action target."""


@lru_cache(maxsize=None)
def _registry(path: str) -> Mapping[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"unsupported action-backend registry: {path}")
    for trajectory_id, spec in raw["action_targets"].items():
        if spec.get("target") not in TARGETS:
            raise ValueError(f"{trajectory_id}: unknown action target {spec.get('target')!r}")
    for name, backend in raw["backends"].items():
        declared = set(backend.get("supports", {})) | set(backend.get("unsupported", {}))
        if declared != set(TARGETS):
            raise ValueError(f"backend {name} must classify every target {TARGETS}; got {sorted(declared)}")
    return raw


def registry(path: Path = REGISTRY_PATH) -> Mapping[str, Any]:
    return _registry(str(path))


def backend(name: str, *, path: Path = REGISTRY_PATH) -> Mapping[str, Any]:
    try:
        return registry(path)["backends"][name]
    except KeyError as error:
        raise KeyError(f"unknown action backend: {name}") from error


def action_target(trajectory: Mapping[str, Any], *, path: Path = REGISTRY_PATH) -> str:
    """Return the entity a semantic trajectory actuates."""
    trajectory_id = str(trajectory["trajectory_id"])
    try:
        return str(registry(path)["action_targets"][trajectory_id]["target"])
    except KeyError as error:
        raise KeyError(f"no action target declared for trajectory {trajectory_id}") from error


def support_status(name: str, target: str, *, path: Path = REGISTRY_PATH) -> str | None:
    """Support status string, or ``None`` when the backend cannot express ``target``."""
    status = backend(name, path=path).get("supports", {}).get(target)
    return None if status is None else str(status)


def require_support(name: str, trajectory: Mapping[str, Any], *, path: Path = REGISTRY_PATH) -> str:
    """Return the trajectory's target, or raise ``UnsupportedAction`` with the recorded reason."""
    target = action_target(trajectory, path=path)
    if support_status(name, target, path=path) is None:
        reason = backend(name, path=path).get("unsupported", {}).get(target, "not supported")
        raise UnsupportedAction(f"{name} cannot express {target} actions ({trajectory['trajectory_id']}): {reason}")
    return target


def camera_gain(name: str, case_id: str, integrated_yaw_deg: float, *, path: Path = REGISTRY_PATH) -> tuple[float, str]:
    """Frozen visual degrees per mouse unit per frame, and the calibration key used."""
    gains = backend(name, path=path)["camera_gain_deg_per_mouse_unit"]
    if "pooled_bidirectional" in gains:
        return float(gains["pooled_bidirectional"]), "pooled_bidirectional"
    regime = "perturb" if "/perturb_" in case_id else "full_turn"
    direction = "negative" if integrated_yaw_deg < 0 else "positive"
    key = f"{regime}_{direction}"
    return float(gains[key]), key
