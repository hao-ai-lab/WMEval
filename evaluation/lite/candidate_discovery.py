from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .case_loader import ALLOWED_CASES


_MANIFEST_FIELD_ORDER = ("model", "case_id", "seed", "candidate_video")
_EMPTY_PARAMS: Mapping[str, Any] = MappingProxyType({})


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True, order=True)
class CandidateRow:
    model: str
    case_id: str
    seed: str
    candidate_video: Path
    # Optional per-candidate parameters (e.g. shadow_hud_preset). Excluded
    # from identity/ordering: (model, case_id, seed) stays the identity.
    params: Mapping[str, Any] = field(default=_EMPTY_PARAMS, compare=False)


def _add_unique(
    rows: list[CandidateRow],
    identities: set[tuple[str, str, str]],
    row: CandidateRow,
    where: str,
) -> None:
    identity = (row.model, row.case_id, row.seed)
    if identity in identities:
        raise ValueError(f"{where}: duplicate candidate identity {identity!r}")
    identities.add(identity)
    rows.append(row)


def _safe_identity_component(value: Any, field: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {field} must be a nonblank string")
    if (
        value in {".", ".."}
        or Path(value).is_absolute()
        or "/" in value
        or "\\" in value
        or "\0" in value
    ):
        raise ValueError(
            f"{where}: {field} must be a safe single filesystem path component, "
            f"got {value!r}"
        )
    return value


def discover_candidate_directory(
    model_root: str | Path,
    *,
    allowed_cases: Iterable[str] | None = None,
) -> list[CandidateRow]:
    root = Path(model_root)
    model = _safe_identity_component(root.name, "model", f"model root {root}")
    rows: list[CandidateRow] = []
    identities: set[tuple[str, str, str]] = set()

    case_ids = ALLOWED_CASES if allowed_cases is None else frozenset(allowed_cases)
    for case_id in sorted(case_ids):
        for video_path in root.glob(f"{case_id}/*.mp4"):
            seed = _safe_identity_component(
                video_path.stem, "seed", f"candidate video {video_path}"
            )
            row = CandidateRow(
                model=model,
                case_id=case_id,
                seed=seed,
                candidate_video=video_path.resolve(),
            )
            _add_unique(rows, identities, row, str(video_path))

    return sorted(rows)


def _manifest_error(path: Path, line_number: int, message: str) -> ValueError:
    return ValueError(f"{path}: line {line_number}: {message}")


def _nonblank_manifest_string(
    value: Any, field: str, path: Path, line_number: int
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _manifest_error(path, line_number, f"{field} must be a nonblank string")
    return value


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_candidate_manifest(
    path: str | Path,
    *,
    allowed_cases: Iterable[str] | None = None,
) -> list[CandidateRow]:
    manifest_path = Path(path).resolve()
    allowed = ALLOWED_CASES if allowed_cases is None else frozenset(allowed_cases)
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"cannot read candidate manifest {manifest_path}: {exc}"
        ) from exc

    rows: list[CandidateRow] = []
    identities: set[tuple[str, str, str]] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line, object_pairs_hook=_strict_json_object)
        except _DuplicateJsonKey as exc:
            raise _manifest_error(manifest_path, line_number, str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise _manifest_error(
                manifest_path, line_number, f"invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(raw, dict):
            raise _manifest_error(
                manifest_path, line_number, "row must be a JSON object"
            )

        for field in _MANIFEST_FIELD_ORDER:
            if field not in raw:
                raise _manifest_error(
                    manifest_path, line_number, f"missing required field {field!r}"
                )

        where = f"{manifest_path}: line {line_number}"
        model = _safe_identity_component(
            raw["model"], "model", where
        )
        case_id = _nonblank_manifest_string(
            raw["case_id"], "case_id", manifest_path, line_number
        )
        seed = _safe_identity_component(
            raw["seed"], "seed", where
        )
        candidate_video_value = _nonblank_manifest_string(
            raw["candidate_video"], "candidate_video", manifest_path, line_number
        )
        if case_id not in allowed:
            raise _manifest_error(
                manifest_path,
                line_number,
                f"unsupported official Lite case_id {case_id!r}",
            )
        params: Mapping[str, Any] = _EMPTY_PARAMS
        if "params" in raw:
            params_value = raw["params"]
            if not isinstance(params_value, dict) or any(
                not isinstance(key, str) for key in params_value
            ):
                raise _manifest_error(
                    manifest_path,
                    line_number,
                    "params must be a JSON object with string keys",
                )
            params = MappingProxyType(dict(params_value))

        candidate_video = Path(candidate_video_value)
        if not candidate_video.is_absolute():
            candidate_video = manifest_path.parent / candidate_video
        row = CandidateRow(model, case_id, seed, candidate_video.resolve(), params)
        _add_unique(
            rows,
            identities,
            row,
            f"{manifest_path}: line {line_number}",
        )

    return rows
