"""Parse and validate ``suite.yaml`` — the dimension-level suite manifest.

The suite manifest is the single source of truth for *what* the benchmark
contains: which dimensions exist, which evaluator drives each dimension
(``registry`` → :func:`evaluation.runner.run_world_model_evaluation`;
``vlm_consistency`` → the prompt-consistency judge), which cases each
dimension runs on, its metric configuration, and how its per-case metrics
aggregate into a headline score.

Case-level ground truth (reference bundle, protocol, evaluator lock) stays in
each case's ``case.yaml`` (see :mod:`evaluation.lite.case_loader`); this
module only records *references* to cases and validates that the referenced
case directories exist.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ._validate import as_mapping, nonblank_str, require_key
from .case_loader import _UniqueKeySafeLoader

EVALUATOR_REGISTRY = "registry"
EVALUATOR_VLM_CONSISTENCY = "vlm_consistency"
_KNOWN_EVALUATORS = (EVALUATOR_REGISTRY, EVALUATOR_VLM_CONSISTENCY)

_AGGREGATE_DIRECTIONS = ("lower", "higher")

_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})

# Default VLM judge settings for vlm_consistency dimensions; ``model`` is
# deliberately required (no default) so the judge model is always pinned by
# the suite manifest rather than drifting with provider defaults.
# ``rubric_dir`` is resolved against each CASE root (rubrics ship inside the
# checksummed case bundle).
_VLM_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "provider": "openai",
        "prompt_version": "v1",
        "alpha": 0.6,
        "rubric_dir": "reference/artifacts/rubrics",
    }
)


class SuiteValidationError(ValueError):
    pass


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    return require_key(mapping, key, where, SuiteValidationError)


def _mapping_value(value: Any, where: str) -> Mapping[str, Any]:
    return as_mapping(value, where, SuiteValidationError)


def _nonblank(value: Any, where: str) -> str:
    return nonblank_str(value, where, SuiteValidationError)


@dataclass(frozen=True)
class AggregateSpec:
    headline: str
    direction: str
    extra_means: tuple[str, ...] = ()


@dataclass(frozen=True)
class DimensionSpec:
    dimension_id: str
    display_name: str
    description: str
    evaluator: str
    requires_reference: bool
    cases: tuple[str, ...]
    modules: tuple[str, ...] = ()
    metric_config: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_MAPPING
    )
    vlm: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_MAPPING)
    aggregate: AggregateSpec | None = None


@dataclass(frozen=True)
class Suite:
    root: Path
    suite_name: str
    version: str
    description: str
    dimensions: Mapping[str, DimensionSpec]
    legacy_case_aliases: Mapping[str, Mapping[str, str]]
    # Repeats per (model, case) a dimension headline requires (suite.yaml
    # `repeat_policy.required_repeats`); None disables the coverage gate.
    required_repeats: int | None = None

    @property
    def cases_root(self) -> Path:
        return self.root / "cases"

    def all_case_ids(self) -> frozenset[str]:
        return frozenset(
            case_id
            for dimension in self.dimensions.values()
            for case_id in dimension.cases
        )

    def dimensions_for_case(self, case_id: str) -> tuple[str, ...]:
        return tuple(
            dimension_id
            for dimension_id, dimension in self.dimensions.items()
            if case_id in dimension.cases
        )


def _parse_case_id(value: Any, where: str) -> str:
    case_id = _nonblank(value, where)
    parts = case_id.split("/")
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise SuiteValidationError(
            f"{where} must be a '<scene>/<variant>' case id, got {case_id!r}"
        )
    return case_id


def _parse_aggregate(raw: Any, where: str) -> AggregateSpec:
    mapping = _mapping_value(raw, where)
    headline = _nonblank(_required(mapping, "headline", where), f"{where}.headline")
    direction = _nonblank(
        _required(mapping, "direction", where), f"{where}.direction"
    )
    if direction not in _AGGREGATE_DIRECTIONS:
        raise SuiteValidationError(
            f"{where}.direction must be one of {_AGGREGATE_DIRECTIONS}, got {direction!r}"
        )
    extra_raw = mapping.get("extra_means", [])
    if not isinstance(extra_raw, list):
        raise SuiteValidationError(f"{where}.extra_means must be a list")
    extra_means = tuple(
        _nonblank(item, f"{where}.extra_means[{index}]")
        for index, item in enumerate(extra_raw)
    )
    return AggregateSpec(headline=headline, direction=direction, extra_means=extra_means)


def _parse_dimension(
    dimension_id: str,
    raw: Any,
    *,
    metric_config_keys: frozenset[str] | None,
) -> DimensionSpec:
    where = f"dimensions.{dimension_id}"
    mapping = _mapping_value(raw, where)

    evaluator = _nonblank(_required(mapping, "evaluator", where), f"{where}.evaluator")
    if evaluator not in _KNOWN_EVALUATORS:
        raise SuiteValidationError(
            f"{where}.evaluator must be one of {_KNOWN_EVALUATORS}, got {evaluator!r}"
        )

    requires_reference = _required(mapping, "requires_reference", where)
    if not isinstance(requires_reference, bool):
        raise SuiteValidationError(f"{where}.requires_reference must be a boolean")

    cases_raw = _required(mapping, "cases", where)
    if not isinstance(cases_raw, list) or not cases_raw:
        raise SuiteValidationError(f"{where}.cases must be a non-empty list")
    cases = tuple(
        _parse_case_id(item, f"{where}.cases[{index}]")
        for index, item in enumerate(cases_raw)
    )
    if len(set(cases)) != len(cases):
        raise SuiteValidationError(f"{where}.cases contains duplicates")

    modules: tuple[str, ...] = ()
    metric_config: Mapping[str, Any] = _EMPTY_MAPPING
    vlm: Mapping[str, Any] = _EMPTY_MAPPING

    if evaluator == EVALUATOR_REGISTRY:
        modules_raw = _required(mapping, "modules", where)
        if not isinstance(modules_raw, list) or not modules_raw:
            raise SuiteValidationError(f"{where}.modules must be a non-empty list")
        modules = tuple(
            _nonblank(item, f"{where}.modules[{index}]")
            for index, item in enumerate(modules_raw)
        )
        raw_config = mapping.get("metric_config", {})
        config = dict(_mapping_value(raw_config, f"{where}.metric_config"))
        for key in config:
            if not isinstance(key, str) or not key.strip():
                raise SuiteValidationError(
                    f"{where}.metric_config keys must be nonblank strings"
                )
        if metric_config_keys is not None:
            unknown = sorted(set(config) - metric_config_keys)
            if unknown:
                raise SuiteValidationError(
                    f"{where}.metric_config has keys outside the sanctioned runner "
                    f"whitelist: {unknown}. Extend METRIC_CONFIG_KEYS deliberately "
                    f"if the runner supports them."
                )
        metric_config = MappingProxyType(config)
        if "vlm" in mapping:
            raise SuiteValidationError(
                f"{where}: 'vlm' is only valid for evaluator=vlm_consistency"
            )
    else:  # vlm_consistency
        if "modules" in mapping or "metric_config" in mapping:
            raise SuiteValidationError(
                f"{where}: 'modules'/'metric_config' are only valid for "
                f"evaluator=registry"
            )
        vlm_raw = dict(_mapping_value(_required(mapping, "vlm", where), f"{where}.vlm"))
        merged = dict(_VLM_DEFAULTS)
        merged.update(vlm_raw)
        _nonblank(merged.get("provider"), f"{where}.vlm.provider")
        _nonblank(
            require_key(merged, "model", f"{where}.vlm", SuiteValidationError),
            f"{where}.vlm.model",
        )
        _nonblank(merged.get("prompt_version"), f"{where}.vlm.prompt_version")
        _nonblank(merged.get("rubric_dir"), f"{where}.vlm.rubric_dir")
        alpha = merged.get("alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise SuiteValidationError(f"{where}.vlm.alpha must be a number")
        if not 0.0 <= float(alpha) <= 1.0:
            raise SuiteValidationError(f"{where}.vlm.alpha must be within [0, 1]")
        vlm = MappingProxyType(merged)

    aggregate: AggregateSpec | None = None
    if "aggregate" in mapping:
        aggregate = _parse_aggregate(mapping["aggregate"], f"{where}.aggregate")

    return DimensionSpec(
        dimension_id=dimension_id,
        display_name=str(mapping.get("display_name") or dimension_id),
        description=str(mapping.get("description") or ""),
        evaluator=evaluator,
        requires_reference=requires_reference,
        cases=cases,
        modules=modules,
        metric_config=metric_config,
        vlm=vlm,
        aggregate=aggregate,
    )


def _required_repeats(raw: Mapping[str, Any]) -> int | None:
    policy = raw.get("repeat_policy")
    if policy is None:
        return None
    policy = _mapping_value(policy, "repeat_policy")
    value = policy.get("required_repeats")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SuiteValidationError("repeat_policy.required_repeats must be a positive integer")
    return value


def load_suite(
    suite_root: str | Path,
    *,
    metric_config_keys: frozenset[str] | None = None,
    check_case_dirs: bool = True,
) -> Suite:
    """Load and validate ``<suite_root>/suite.yaml``.

    ``metric_config_keys`` defaults to the registry adapter's sanctioned
    runner-kwarg whitelist; pass ``None`` explicitly via the keyword to skip
    that check (tests only).
    """
    if metric_config_keys is None:
        from .adapters.registry_adapter import METRIC_CONFIG_KEYS

        metric_config_keys = METRIC_CONFIG_KEYS

    root = Path(suite_root).resolve()
    manifest_path = root / "suite.yaml"
    try:
        raw = yaml.load(
            manifest_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
        )
    except OSError as exc:
        raise SuiteValidationError(f"cannot read {manifest_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SuiteValidationError(
            f"invalid YAML in {manifest_path}: {exc}"
        ) from exc

    raw = _mapping_value(raw, str(manifest_path))
    suite_name = _nonblank(_required(raw, "suite_name", "suite.yaml"), "suite_name")
    version = _nonblank(_required(raw, "version", "suite.yaml"), "version")
    description = str(raw.get("description") or "")

    dimensions_raw = _mapping_value(
        _required(raw, "dimensions", "suite.yaml"), "dimensions"
    )
    if not dimensions_raw:
        raise SuiteValidationError("suite.yaml requires a non-empty dimensions mapping")
    dimensions: dict[str, DimensionSpec] = {}
    for dimension_id, payload in dimensions_raw.items():
        dimension_id = _nonblank(dimension_id, "dimensions key")
        dimensions[dimension_id] = _parse_dimension(
            dimension_id, payload, metric_config_keys=metric_config_keys
        )

    aliases_raw = raw.get("legacy_case_aliases", {})
    aliases_mapping = _mapping_value(aliases_raw, "legacy_case_aliases")
    aliases: dict[str, Mapping[str, str]] = {}
    for dimension_id, alias_map_raw in aliases_mapping.items():
        if dimension_id not in dimensions:
            raise SuiteValidationError(
                f"legacy_case_aliases references unknown dimension {dimension_id!r}"
            )
        alias_map = _mapping_value(
            alias_map_raw, f"legacy_case_aliases.{dimension_id}"
        )
        resolved: dict[str, str] = {}
        for legacy_id, target in alias_map.items():
            legacy_id = _nonblank(
                legacy_id, f"legacy_case_aliases.{dimension_id} key"
            )
            target = _nonblank(
                target, f"legacy_case_aliases.{dimension_id}.{legacy_id}"
            )
            if target not in dimensions[dimension_id].cases:
                raise SuiteValidationError(
                    f"legacy_case_aliases.{dimension_id}.{legacy_id} targets "
                    f"{target!r} which is not one of that dimension's cases"
                )
            resolved[legacy_id] = target
        aliases[dimension_id] = MappingProxyType(resolved)

    suite = Suite(
        root=root,
        suite_name=suite_name,
        version=version,
        description=description,
        dimensions=MappingProxyType(dimensions),
        legacy_case_aliases=MappingProxyType(aliases),
        required_repeats=_required_repeats(raw),
    )

    if check_case_dirs:
        missing = sorted(
            case_id
            for case_id in suite.all_case_ids()
            if not (suite.cases_root / case_id / "case.yaml").is_file()
        )
        if missing:
            raise SuiteValidationError(
                f"suite.yaml references cases without a case.yaml under "
                f"{suite.cases_root}: {missing}"
            )

    return suite
