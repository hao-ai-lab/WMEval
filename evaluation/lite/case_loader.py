from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


ALLOWED_CASES = frozenset(
    {
        "beamdomino/canonical",
        "beamdomino/glancing",
        "beamdomino/direct_strike",
        "liquid_container/rest",
        "liquid_container/step_hold",
        "liquid_container/impulse_decay",
        "liquid_container/chirp_strafe",
    }
)

SCENE_MODULES = {
    "beamdomino": ("causal_step",),
    "liquid_container": (
        "liquid_velocity",
        "liquid_volume_conservation",
        "liquid_surface_orientation",
        "slosh_spectrum",
    ),
}

_PROTOCOL_KEYS = (
    "candidate_normalization",
    "min_duration_sec",
    "min_duration_ratio",
    "duration_threshold_mode",
)
_EVALUATOR_LOCK_KEYS = (
    "release_id",
    "code_revision",
    "evaluator_fingerprint",
    "module_config",
    "artifact_schema_versions",
    "dependency_versions",
)
_EVALUATOR_LOCK_MAPPING_KEYS = (
    "module_config",
    "artifact_schema_versions",
    "dependency_versions",
)


@dataclass(frozen=True)
class LiteCase:
    root: Path
    schema_version: int
    case_id: str
    scene: str
    display_name: str
    modules: tuple[str, ...]
    reference_video: Path
    baseline_path: Path
    checksums_path: Path
    protocol: Mapping[str, Any]
    evaluator_lock: Mapping[str, Any]

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "reference" / "artifacts"


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{where} missing required key {key!r}")
    return mapping[key]


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain a mapping")
    return value


def _require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...], where: str) -> None:
    for key in keys:
        _required(mapping, key, where)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def load_case(case_dir: str | Path) -> LiteCase:
    root = Path(case_dir).resolve()
    manifest_path = root / "case.yaml"
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {manifest_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {manifest_path}: {exc}") from exc

    raw = _mapping(raw, str(manifest_path))
    case_id = str(_required(raw, "case_id", "case.yaml"))
    scene = str(_required(raw, "scene", "case.yaml"))
    modules_value = _required(raw, "modules", "case.yaml")
    if not isinstance(modules_value, list):
        raise ValueError("case.yaml key 'modules' must be a list")
    modules = tuple(str(value) for value in modules_value)

    if case_id not in ALLOWED_CASES:
        raise ValueError(f"unsupported official Lite case_id {case_id!r}")
    expected_scene = case_id.partition("/")[0]
    if scene != expected_scene:
        raise ValueError(f"{case_id}: case_id does not match scene {scene!r}")
    expected_modules = SCENE_MODULES[scene]
    if modules != expected_modules:
        raise ValueError(f"{case_id}: modules must be {expected_modules!r}")

    reference = _mapping(_required(raw, "reference", "case.yaml"), "reference")
    protocol = _mapping(_required(raw, "protocol", "case.yaml"), "protocol")
    evaluator_lock = _mapping(
        _required(raw, "evaluator_lock", "case.yaml"), "evaluator_lock"
    )
    _require_keys(protocol, _PROTOCOL_KEYS, "protocol")
    _require_keys(evaluator_lock, _EVALUATOR_LOCK_KEYS, "evaluator_lock")
    for key in _EVALUATOR_LOCK_MAPPING_KEYS:
        _mapping(evaluator_lock[key], f"evaluator_lock.{key}")

    return LiteCase(
        root=root,
        schema_version=int(_required(raw, "schema_version", "case.yaml")),
        case_id=case_id,
        scene=scene,
        display_name=str(_required(raw, "display_name", "case.yaml")),
        modules=modules,
        reference_video=(root / str(_required(reference, "video", "reference"))).resolve(),
        baseline_path=(root / str(_required(reference, "baseline", "reference"))).resolve(),
        checksums_path=(root / str(_required(reference, "checksums", "reference"))).resolve(),
        protocol=_freeze(protocol),
        evaluator_lock=_freeze(evaluator_lock),
    )


def discover_cases(cases_root: str | Path) -> list[LiteCase]:
    root = Path(cases_root)
    return sorted(
        (load_case(path.parent) for path in root.glob("*/*/case.yaml")),
        key=lambda case: case.case_id,
    )
