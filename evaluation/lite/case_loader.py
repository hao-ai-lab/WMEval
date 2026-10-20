from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from ._validate import as_mapping, nonblank_str, positive_number, require_key


_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})

# module_params keys whose value is a Unity ground-truth artifact path,
# authored relative to the case bundle directory -- resolved to absolute at
# load time in load_case(), the same way reference.artifacts.* already is.
_MODULE_PARAM_PATH_KEYS = frozenset(
    {
        "eoc_unity_object_transforms_json",
        "eoc_unity_poses_npy",
        "eoc_unity_intrinsics_npy",
        "eoc_unity_object_diameters_json",
        "kin_unity_object_transforms_json",
        "kin_unity_poses_npy",
        "kin_unity_intrinsics_npy",
        "kin_unity_object_masses_json",
        "lpips_reference_events_json",
        "kin_plane_calibration_json",
    }
)

# Hard-coded v1 registry. schema_version 1 cases are always checked against
# these; schema_version 2 cases are checked against whatever the caller (the
# suite layer) passes via ``allowed_cases`` / ``expected_modules`` instead.
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
_EVALUATOR_LOCK_SCALAR_KEYS = (
    "release_id",
    "code_revision",
    "evaluator_fingerprint",
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate mapping key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class LiteCase:
    root: Path
    schema_version: int
    case_id: str
    scene: str
    display_name: str
    modules: tuple[str, ...]
    reference_video: Path | None
    # Required for schema_version 1; optional in v2 (a baseline file exists
    # only when real baseline scores exist — no `{}` placeholders).
    baseline_path: Path | None
    checksums_path: Path
    protocol: Mapping[str, Any]
    evaluator_lock: Mapping[str, Any]
    # schema_version 2 additions; v1 cases keep the defaults.
    prompt: str | None = None
    perspective: str | None = None
    rubric_scene_name: str | None = None
    first_frame: Path | None = None
    module_params: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_MAPPING)
    # Named auxiliary reference assets (e.g. synthetic_action_dir, actions_file,
    # qa_dataset), resolved to absolute paths under the case root.
    reference_artifacts: Mapping[str, Path] = field(
        default_factory=lambda: _EMPTY_MAPPING
    )
    # Set for synthetic trajectories (evaluation.lite.synthetic), whose action
    # target is declared in the manifest rather than in reference/artifacts.
    action_target: str | None = None
    # Directory the vlm_consistency ``rubric_dir`` resolves against; None means
    # ``root``. Synthetic trajectories reuse their conditioning case's rubric.
    rubric_root: Path | None = None

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "reference" / "artifacts"


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    return require_key(mapping, key, where, ValueError)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    return as_mapping(value, where, ValueError)


def _require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...], where: str) -> None:
    for key in keys:
        _required(mapping, key, where)


def _nonblank_string(value: Any, where: str) -> str:
    return nonblank_str(value, where, ValueError)


def _positive_number(value: Any, where: str) -> int | float:
    return positive_number(value, where, ValueError)


def _freeze(value: Any, where: str) -> Any:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{where} mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze(item, f"{where}.{key}") for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(
            _freeze(item, f"{where}[{index}]") for index, item in enumerate(value)
        )
    if isinstance(value, set):
        return frozenset(_freeze(item, where) for item in value)
    return value


def load_case(
    case_dir: str | Path,
    *,
    allowed_cases: Iterable[str] | None = None,
    expected_modules: Mapping[str, tuple[str, ...]] | None = None,
) -> LiteCase:
    """Parse and validate one ``case.yaml``.

    ``allowed_cases`` / ``expected_modules`` default to the hard-coded v1
    registries for ``schema_version: 1`` cases (preserving the original strict
    behaviour). For ``schema_version: 2`` cases they default to ``None``,
    meaning the caller (the suite layer, which derives them from
    ``suite.yaml``) is responsible for membership checks.
    """
    root = Path(case_dir).resolve()
    manifest_path = root / "case.yaml"
    try:
        raw = yaml.load(
            manifest_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
        )
    except OSError as exc:
        raise ValueError(f"cannot read {manifest_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {manifest_path}: {exc}") from exc

    raw = _mapping(raw, str(manifest_path))
    schema_version = _required(raw, "schema_version", "case.yaml")
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError("schema_version must be integer 1 or 2")
    case_id = _nonblank_string(_required(raw, "case_id", "case.yaml"), "case_id")
    scene = _nonblank_string(_required(raw, "scene", "case.yaml"), "scene")
    display_name = _nonblank_string(
        _required(raw, "display_name", "case.yaml"), "display_name"
    )
    modules_value = _required(raw, "modules", "case.yaml")
    if not isinstance(modules_value, list):
        raise ValueError("case.yaml key 'modules' must be a list")
    if any(not isinstance(value, str) or not value.strip() for value in modules_value):
        raise ValueError("modules entries must be nonblank strings")
    modules = tuple(modules_value)

    resolved_allowed = (
        frozenset(allowed_cases)
        if allowed_cases is not None
        else (ALLOWED_CASES if schema_version == 1 else None)
    )
    if resolved_allowed is not None and case_id not in resolved_allowed:
        raise ValueError(f"unsupported official Lite case_id {case_id!r}")
    expected_scene = case_id.partition("/")[0]
    if scene != expected_scene:
        raise ValueError(f"{case_id}: case_id does not match scene {scene!r}")
    resolved_modules = (
        expected_modules
        if expected_modules is not None
        else (SCENE_MODULES if schema_version == 1 else None)
    )
    if resolved_modules is not None:
        expected = resolved_modules.get(scene)
        if expected is None:
            raise ValueError(f"{case_id}: no expected modules for scene {scene!r}")
        if modules != tuple(expected):
            raise ValueError(f"{case_id}: modules must be {tuple(expected)!r}")

    reference = _mapping(_required(raw, "reference", "case.yaml"), "reference")
    protocol = _mapping(_required(raw, "protocol", "case.yaml"), "protocol")
    evaluator_lock = _mapping(
        _required(raw, "evaluator_lock", "case.yaml"), "evaluator_lock"
    )
    _require_keys(protocol, _PROTOCOL_KEYS, "protocol")
    _require_keys(evaluator_lock, _EVALUATOR_LOCK_KEYS, "evaluator_lock")
    _nonblank_string(
        protocol["candidate_normalization"], "protocol.candidate_normalization"
    )
    _positive_number(protocol["min_duration_sec"], "protocol.min_duration_sec")
    _positive_number(protocol["min_duration_ratio"], "protocol.min_duration_ratio")
    if protocol["duration_threshold_mode"] != "max":
        raise ValueError("protocol.duration_threshold_mode must be exactly 'max'")
    for key in _EVALUATOR_LOCK_SCALAR_KEYS:
        _nonblank_string(evaluator_lock[key], f"evaluator_lock.{key}")
    for key in _EVALUATOR_LOCK_MAPPING_KEYS:
        _mapping(evaluator_lock[key], f"evaluator_lock.{key}")

    # reference.video is required for schema_version 1; optional for
    # schema_version 2 so reference-free dimensions (prompt/shadow
    # consistency) can ship a bundle with only auxiliary reference assets.
    reference_video: str | None
    if schema_version == 1 or "video" in reference:
        reference_video = _nonblank_string(
            _required(reference, "video", "reference"), "reference.video"
        )
    else:
        reference_video = None
    first_frame: str | None = None
    if schema_version >= 2 and "first_frame" in reference:
        first_frame = _nonblank_string(
            reference["first_frame"], "reference.first_frame"
        )
    baseline_path: str | None
    if schema_version == 1 or "baseline" in reference:
        baseline_path = _nonblank_string(
            _required(reference, "baseline", "reference"), "reference.baseline"
        )
    else:
        baseline_path = None
    checksums_path = _nonblank_string(
        _required(reference, "checksums", "reference"), "reference.checksums"
    )
    reference_artifacts: Mapping[str, Path] = _EMPTY_MAPPING
    if schema_version >= 2 and "artifacts" in reference:
        artifacts_raw = _mapping(reference["artifacts"], "reference.artifacts")
        resolved_artifacts: dict[str, Path] = {}
        for name, value in artifacts_raw.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("reference.artifacts keys must be nonblank strings")
            relative = _nonblank_string(value, f"reference.artifacts.{name}")
            resolved_artifacts[name] = (root / relative).resolve()
        reference_artifacts = MappingProxyType(resolved_artifacts)

    prompt: str | None = None
    perspective: str | None = None
    rubric_scene_name: str | None = None
    module_params: Mapping[str, Any] = _EMPTY_MAPPING
    if schema_version >= 2:
        if "prompt" in raw:
            prompt = _nonblank_string(raw["prompt"], "prompt")
        if "perspective" in raw:
            perspective = _nonblank_string(raw["perspective"], "perspective")
        if "rubric_scene_name" in raw:
            rubric_scene_name = _nonblank_string(
                raw["rubric_scene_name"], "rubric_scene_name"
            )
        if "module_params" in raw:
            module_params_raw = _mapping(raw["module_params"], "module_params")
            resolved_module_params: dict[str, Any] = {}
            for module_name, params in module_params_raw.items():
                params_map = as_mapping(
                    params, f"module_params.{module_name}", ValueError
                )
                # Unity-ground-truth artifact paths (kin_unity_*/eoc_unity_*)
                # are authored in case.yaml relative to the case bundle
                # directory, exactly like reference.artifacts.* just above --
                # but unlike reference.artifacts.*, nothing was resolving
                # them to absolute paths, so any case relying on them (every
                # environment_object_consistency/kinematics_mechanics case in
                # the suite) would fail with a spurious "file does not exist"
                # the moment it's run from any working directory other than
                # the case's own bundle dir. (root / value) is a no-op for an
                # already-absolute value, matching reference.artifacts's own
                # (root / relative).resolve() convention.
                resolved_params = dict(params_map)
                for key in _MODULE_PARAM_PATH_KEYS:
                    value = resolved_params.get(key)
                    if isinstance(value, str) and value.strip():
                        resolved_params[key] = str((root / value).resolve())
                resolved_module_params[module_name] = resolved_params
            module_params = _freeze(resolved_module_params, "module_params")

    return LiteCase(
        root=root,
        schema_version=schema_version,
        case_id=case_id,
        scene=scene,
        display_name=display_name,
        modules=modules,
        reference_video=(
            (root / reference_video).resolve() if reference_video is not None else None
        ),
        baseline_path=(
            (root / baseline_path).resolve() if baseline_path is not None else None
        ),
        checksums_path=(root / checksums_path).resolve(),
        protocol=_freeze(protocol, "protocol"),
        evaluator_lock=_freeze(evaluator_lock, "evaluator_lock"),
        prompt=prompt,
        perspective=perspective,
        rubric_scene_name=rubric_scene_name,
        first_frame=(
            (root / first_frame).resolve() if first_frame is not None else None
        ),
        module_params=module_params,
        reference_artifacts=reference_artifacts,
    )


def discover_cases(
    cases_root: str | Path,
    *,
    allowed_cases: Iterable[str] | None = None,
    expected_modules: Mapping[str, tuple[str, ...]] | None = None,
) -> list[LiteCase]:
    root = Path(cases_root)
    return sorted(
        (
            load_case(
                path.parent,
                allowed_cases=allowed_cases,
                expected_modules=expected_modules,
            )
            for path in root.glob("*/*/case.yaml")
        ),
        key=lambda case: case.case_id,
    )
