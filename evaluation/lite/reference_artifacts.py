"""Portable, pickle-free reference artifacts for WM-Eval Lite liquid cases."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from evaluation.modules._liquid_common import (
    BottleTrace,
    LiquidMaskResult,
    PIVField,
    store_liquid_mask_cache,
    store_raft_flow_cache,
)

from .case_loader import LiteCase


class ReferenceArtifactError(ValueError):
    """A portable Lite reference artifact or its lock metadata is invalid."""


def _artifact_error(kind: str, path: Path, detail: str) -> ReferenceArtifactError:
    return ReferenceArtifactError(f"{kind} artifact {path}: {detail}")


def _unicode_scalar(value: str) -> np.ndarray:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _write(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _read_arrays(path: Path, kind: str, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in required if key not in archive.files]
            if missing:
                raise _artifact_error(kind, path, f"missing required keys {missing!r}")
            return {key: archive[key] for key in required}
    except ReferenceArtifactError:
        raise
    except (OSError, ValueError, BadZipFile) as exc:
        raise _artifact_error(kind, path, f"cannot load safely: {exc}") from exc


def _require_array(
    array: np.ndarray,
    *,
    kind: str,
    path: Path,
    key: str,
    dtype: np.dtype[Any] | type,
    ndim: int,
    trailing_shape: tuple[int, ...] = (),
) -> None:
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype:
        raise _artifact_error(
            kind, path, f"{key} dtype must be {expected_dtype}, got {array.dtype}"
        )
    if array.ndim != ndim:
        raise _artifact_error(kind, path, f"{key} rank must be {ndim}, got {array.ndim}")
    if trailing_shape and array.shape[-len(trailing_shape) :] != trailing_shape:
        raise _artifact_error(
            kind,
            path,
            f"{key} trailing shape must be {trailing_shape}, got {array.shape}",
        )


def _require_unicode_scalar(
    array: np.ndarray, *, kind: str, path: Path, key: str
) -> str:
    if array.ndim != 0 or array.dtype.kind != "U":
        raise _artifact_error(
            kind, path, f"{key} must be a fixed-Unicode scalar, got {array.dtype} {array.shape}"
        )
    value = str(array.item())
    if not value:
        raise _artifact_error(kind, path, f"{key} must not be empty")
    return value


def write_liquid_mask(path: Path, result: LiquidMaskResult) -> None:
    kind = "liquid mask"
    _require_array(result.masks, kind=kind, path=path, key="masks", dtype=np.bool_, ndim=3)
    if not math.isfinite(float(result.fps)) or float(result.fps) <= 0:
        raise _artifact_error(kind, path, f"fps must be finite and positive, got {result.fps!r}")
    if not result.sam2_model or not result.prompt:
        raise _artifact_error(kind, path, "sam2_model and prompt must not be empty")
    _write(
        path,
        masks=result.masks,
        fps=np.asarray(result.fps, dtype=np.float64),
        sam2_model=_unicode_scalar(result.sam2_model),
        prompt=_unicode_scalar(result.prompt),
    )


def read_liquid_mask(path: Path) -> LiquidMaskResult:
    kind = "liquid mask"
    arrays = _read_arrays(path, kind, ("masks", "fps", "sam2_model", "prompt"))
    _require_array(arrays["masks"], kind=kind, path=path, key="masks", dtype=np.bool_, ndim=3)
    _require_array(arrays["fps"], kind=kind, path=path, key="fps", dtype=np.float64, ndim=0)
    fps = float(arrays["fps"].item())
    if not math.isfinite(fps) or fps <= 0:
        raise _artifact_error(kind, path, f"fps must be finite and positive, got {fps!r}")
    return LiquidMaskResult(
        masks=arrays["masks"],
        fps=fps,
        sam2_model=_require_unicode_scalar(
            arrays["sam2_model"], kind=kind, path=path, key="sam2_model"
        ),
        prompt=_require_unicode_scalar(arrays["prompt"], kind=kind, path=path, key="prompt"),
    )


def write_bottle_trace(path: Path, result: BottleTrace) -> None:
    kind = "bottle trace"
    _validate_bottle_trace(path, result, kind)
    _write(
        path,
        centroid_xy=result.centroid_xy,
        bbox_xyxy=result.bbox_xyxy,
        valid=result.valid,
    )


def _validate_bottle_trace(path: Path, result: BottleTrace, kind: str) -> None:
    _require_array(
        result.centroid_xy,
        kind=kind,
        path=path,
        key="centroid_xy",
        dtype=np.float64,
        ndim=2,
        trailing_shape=(2,),
    )
    _require_array(
        result.bbox_xyxy,
        kind=kind,
        path=path,
        key="bbox_xyxy",
        dtype=np.float64,
        ndim=2,
        trailing_shape=(4,),
    )
    _require_array(result.valid, kind=kind, path=path, key="valid", dtype=np.bool_, ndim=1)
    frames = result.valid.shape[0]
    if result.centroid_xy.shape[0] != frames or result.bbox_xyxy.shape[0] != frames:
        raise _artifact_error(
            kind,
            path,
            "centroid_xy, bbox_xyxy, and valid must have matching frame counts",
        )


def read_bottle_trace(path: Path) -> BottleTrace:
    kind = "bottle trace"
    arrays = _read_arrays(path, kind, ("centroid_xy", "bbox_xyxy", "valid"))
    result = BottleTrace(
        centroid_xy=arrays["centroid_xy"],
        bbox_xyxy=arrays["bbox_xyxy"],
        valid=arrays["valid"],
    )
    _validate_bottle_trace(path, result, kind)
    return result


def write_piv_field(path: Path, result: PIVField) -> None:
    kind = "PIV field"
    _validate_piv_field(path, result, kind)
    _write(path, u=result.u, v=result.v, valid=result.valid)


def _validate_piv_field(path: Path, result: PIVField, kind: str) -> None:
    _require_array(result.u, kind=kind, path=path, key="u", dtype=np.float32, ndim=3)
    _require_array(result.v, kind=kind, path=path, key="v", dtype=np.float32, ndim=3)
    _require_array(result.valid, kind=kind, path=path, key="valid", dtype=np.bool_, ndim=3)
    if result.u.shape != result.v.shape or result.u.shape != result.valid.shape:
        raise _artifact_error(
            kind,
            path,
            f"u, v, and valid shapes must match, got {result.u.shape}, "
            f"{result.v.shape}, and {result.valid.shape}",
        )


def read_piv_field(path: Path) -> PIVField:
    kind = "PIV field"
    arrays = _read_arrays(path, kind, ("u", "v", "valid"))
    result = PIVField(u=arrays["u"], v=arrays["v"], valid=arrays["valid"])
    _validate_piv_field(path, result, kind)
    return result


@dataclass(frozen=True)
class _LiquidLock:
    fps: float
    sam2_model: str
    liquid_prompt: str
    bottle_prompt: str
    max_pairs: int
    liquid_max_pairs: int | None
    window_size: int
    overlap: int
    min_mask_frac: float


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferenceArtifactError(f"{where} must contain a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ReferenceArtifactError(f"{where} missing required key {key!r}")
    return mapping[key]


def _string(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = _required(mapping, key, where)
    if not isinstance(value, str) or not value:
        raise ReferenceArtifactError(f"{where}.{key} must be a nonempty string")
    return value


def _number(mapping: Mapping[str, Any], key: str, where: str) -> float:
    value = _required(mapping, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceArtifactError(f"{where}.{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ReferenceArtifactError(f"{where}.{key} must be a finite number")
    return number


def _integer(mapping: Mapping[str, Any], key: str, where: str) -> int:
    value = _required(mapping, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReferenceArtifactError(f"{where}.{key} must be an integer")
    return value


def _expected_liquid_lock(case: LiteCase) -> _LiquidLock:
    module_config = _mapping(case.evaluator_lock.get("module_config"), "evaluator_lock.module_config")
    liquid = _mapping(
        _required(module_config, "liquid", "evaluator_lock.module_config"),
        "evaluator_lock.module_config.liquid",
    )
    raft = _mapping(
        _required(liquid, "raft", "evaluator_lock.module_config.liquid"),
        "evaluator_lock.module_config.liquid.raft",
    )
    liquid_max_pairs_raw = _integer(
        liquid, "liquid_max_pairs", "evaluator_lock.module_config.liquid"
    )
    if liquid_max_pairs_raw < 0:
        raise ReferenceArtifactError(
            "evaluator_lock.module_config.liquid.liquid_max_pairs must be nonnegative"
        )
    max_pairs = _integer(liquid, "max_pairs", "evaluator_lock.module_config.liquid")
    if max_pairs <= 0:
        raise ReferenceArtifactError(
            "evaluator_lock.module_config.liquid.max_pairs must be positive"
        )
    window_size = _integer(raft, "window_size", "evaluator_lock.module_config.liquid.raft")
    overlap = _integer(raft, "overlap", "evaluator_lock.module_config.liquid.raft")
    min_mask_frac = _number(raft, "min_mask_frac", "evaluator_lock.module_config.liquid.raft")
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ReferenceArtifactError(
            "evaluator_lock.module_config.liquid.raft requires "
            "window_size > overlap >= 0"
        )
    if not 0.0 <= min_mask_frac <= 1.0:
        raise ReferenceArtifactError(
            "evaluator_lock.module_config.liquid.raft.min_mask_frac must be in [0, 1]"
        )
    fps = _number(liquid, "normalization_fps", "evaluator_lock.module_config.liquid")
    if fps <= 0:
        raise ReferenceArtifactError(
            "evaluator_lock.module_config.liquid.normalization_fps must be positive"
        )
    return _LiquidLock(
        fps=fps,
        sam2_model=_string(liquid, "sam2_model", "evaluator_lock.module_config.liquid"),
        liquid_prompt=_string(liquid, "liquid_prompt", "evaluator_lock.module_config.liquid"),
        bottle_prompt=_string(liquid, "bottle_prompt", "evaluator_lock.module_config.liquid"),
        max_pairs=max_pairs,
        liquid_max_pairs=liquid_max_pairs_raw or None,
        window_size=window_size,
        overlap=overlap,
        min_mask_frac=min_mask_frac,
    )


def _verify_mask_metadata(
    result: LiquidMaskResult, expected: _LiquidLock, *, expected_prompt: str, artifact: str
) -> None:
    expected_values = {
        "fps": expected.fps,
        "sam2_model": expected.sam2_model,
        "prompt": expected_prompt,
    }
    actual_values = {
        "fps": result.fps,
        "sam2_model": result.sam2_model,
        "prompt": result.prompt,
    }
    for key, expected_value in expected_values.items():
        actual_value = actual_values[key]
        if actual_value != expected_value:
            lock_key = "normalization_fps" if key == "fps" else key
            raise ReferenceArtifactError(
                f"{artifact} metadata mismatch for {lock_key}: "
                f"evaluator lock expects {expected_value!r}, artifact has {actual_value!r}"
            )


def install_liquid_reference_artifacts(case: LiteCase, cache_dir: Path) -> None:
    expected = _expected_liquid_lock(case)
    liquid = read_liquid_mask(case.artifacts_dir / "liquid_mask.npz")
    bottle = read_liquid_mask(case.artifacts_dir / "bottle_mask.npz")
    field = read_piv_field(case.artifacts_dir / "raft_pooled_flow.npz")
    _verify_mask_metadata(
        liquid, expected, expected_prompt=expected.liquid_prompt, artifact="liquid_mask.npz"
    )
    _verify_mask_metadata(
        bottle, expected, expected_prompt=expected.bottle_prompt, artifact="bottle_mask.npz"
    )
    store_liquid_mask_cache(
        video_path=case.reference_video,
        result=liquid,
        max_pairs=expected.liquid_max_pairs,
        cache_dir=cache_dir,
    )
    store_liquid_mask_cache(
        video_path=case.reference_video,
        result=bottle,
        max_pairs=expected.liquid_max_pairs,
        cache_dir=cache_dir,
    )
    store_raft_flow_cache(
        video_path=case.reference_video,
        mask=liquid,
        result=field,
        window_size=expected.window_size,
        overlap=expected.overlap,
        fps=expected.fps,
        max_frames=expected.liquid_max_pairs,
        cache_dir=cache_dir,
        min_mask_frac=expected.min_mask_frac,
    )
