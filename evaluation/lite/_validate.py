"""Shared structural-validation primitives for WMEval Lite metadata parsing.

``case_loader`` and ``reference_artifacts`` both parse untrusted YAML/JSON-shaped
metadata and previously kept their own near-identical copies of these checks.
Each primitive here takes an ``error`` factory so a caller can raise its own
exception type (``ValueError`` for case loading, ``ReferenceArtifactError`` for
portable artifacts) with a single, consistent message format, removing the
drift risk between the two copies.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

ErrorFactory = Callable[[str], Exception]


def require_key(
    mapping: Mapping[str, Any], key: str, where: str, error: ErrorFactory
) -> Any:
    if key not in mapping:
        raise error(f"{where} missing required key {key!r}")
    return mapping[key]


def as_mapping(value: Any, where: str, error: ErrorFactory) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{where} must contain a mapping")
    return value


def nonblank_str(value: Any, where: str, error: ErrorFactory) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{where} must be a nonblank string")
    return value


def positive_number(value: Any, where: str, error: ErrorFactory) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        or value <= 0
    ):
        raise error(f"{where} must be a finite positive number")
    return value


def finite_number(value: Any, where: str, error: ErrorFactory) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise error(f"{where} must be a finite number")
    return float(value)


def integer(value: Any, where: str, error: ErrorFactory) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(f"{where} must be an integer")
    return value
