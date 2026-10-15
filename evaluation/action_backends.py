"""Backend registry for compiling canonical action trajectories.

The semantic trajectory YAMLs (``experiments/tables_4_7/trajectories`` and,
for Lite cases outside Tables 4/7, ``experiments/trajectory_reconstructions``)
say *what* happens over time.  ``experiments/tables_4_7/action_backends.yaml`` says
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
TARGETS = ("camera", "ego_vehicle", "cue_ball", "ball", "table_objects", "rolling_ball", "liquid_container")


class UnsupportedAction(ValueError):
    """Raised when a backend cannot express a trajectory's action target."""


class IncompleteAction(ValueError):
    """Capability/adapter has not been verified; this is not an N/A result."""


@lru_cache(maxsize=None)
def _registry(path: str) -> Mapping[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"unsupported action-backend registry: {path}")
    for trajectory_id, spec in raw["action_targets"].items():
        if spec.get("target") not in TARGETS:
            raise ValueError(f"{trajectory_id}: unknown action target {spec.get('target')!r}")
    for name, backend in raw["backends"].items():
        if backend.get("scope") == "trajectories_only":
            if backend.get("supports") or backend.get("unsupported"):
                raise ValueError(f"{name}: scoped backend must not classify unreviewed targets")
            continue
        declared = set(backend.get("supports", {})) | set(backend.get("unsupported", {}))
        if declared != set(TARGETS):
            raise ValueError(f"backend {name} must classify every target {TARGETS}; got {sorted(declared)}")
    for tid, trajectory in raw.get("trajectory_support", {}).items():
        if trajectory.get("target") not in TARGETS:
            raise ValueError(f"{tid}: invalid scoped target")
        for name, decision in trajectory.get("backends", {}).items():
            if name not in raw["backends"] or decision.get("status") not in {"supported", "unsupported", "pending_adapter", "pending_calibration", "blocked_access"}:
                raise ValueError(f"{tid}/{name}: invalid trajectory capability")
            if not decision.get("reason"):
                raise ValueError(f"{tid}/{name}: capability requires a reason")
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
    scoped = registry(path).get("trajectory_support", {}).get(trajectory_id)
    if scoped is not None:
        target = str(scoped["target"])
        if trajectory.get("target", target) != target:
            raise ValueError(f"{trajectory_id}: semantic/registry target mismatch")
        return target
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
    scoped = registry(path).get("trajectory_support", {}).get(trajectory["trajectory_id"])
    if scoped is not None:
        decision = scoped.get("backends", {}).get(name)
        if decision is None:
            raise IncompleteAction(f"{name}: whole trajectory capability not reviewed")
        status = decision["status"]
        if status == "unsupported":
            raise UnsupportedAction(f"{name} cannot express {trajectory['trajectory_id']}: {decision['reason']}")
        if status != "supported":
            raise IncompleteAction(f"{name}/{trajectory['trajectory_id']}: {status}: {decision['reason']}")
        return target
    if trajectory.get("kind") == "camera_look_at_orbit":
        raise IncompleteAction(f"{name}: camera target support alone does not verify a complete orbit")
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
