from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .case_loader import ALLOWED_CASES


_CASE_IDS = (
    "beamdomino/canonical",
    "beamdomino/glancing",
    "beamdomino/direct_strike",
    "liquid_container/rest",
    "liquid_container/step_hold",
    "liquid_container/impulse_decay",
    "liquid_container/chirp_strafe",
)
_MANIFEST_FIELD_ORDER = ("model", "case_id", "seed", "candidate_video")
_MANIFEST_FIELDS = frozenset(_MANIFEST_FIELD_ORDER)


@dataclass(frozen=True, order=True)
class CandidateRow:
    model: str
    case_id: str
    seed: str
    candidate_video: Path


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


def discover_candidate_directory(model_root: str | Path) -> list[CandidateRow]:
    root = Path(model_root)
    model = root.name
    rows: list[CandidateRow] = []
    identities: set[tuple[str, str, str]] = set()

    for case_id in _CASE_IDS:
        for video_path in root.glob(f"{case_id}/*.mp4"):
            row = CandidateRow(
                model=model,
                case_id=case_id,
                seed=video_path.stem,
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


def load_candidate_manifest(path: str | Path) -> list[CandidateRow]:
    manifest_path = Path(path).resolve()
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
            raw = json.loads(line)
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
        unexpected = raw.keys() - _MANIFEST_FIELDS
        if unexpected:
            field = sorted(unexpected)[0]
            raise _manifest_error(
                manifest_path, line_number, f"unexpected field {field!r}"
            )

        model = _nonblank_manifest_string(
            raw["model"], "model", manifest_path, line_number
        )
        case_id = _nonblank_manifest_string(
            raw["case_id"], "case_id", manifest_path, line_number
        )
        seed = _nonblank_manifest_string(
            raw["seed"], "seed", manifest_path, line_number
        )
        candidate_video_value = _nonblank_manifest_string(
            raw["candidate_video"], "candidate_video", manifest_path, line_number
        )
        if case_id not in ALLOWED_CASES:
            raise _manifest_error(
                manifest_path,
                line_number,
                f"unsupported official Lite case_id {case_id!r}",
            )

        candidate_video = Path(candidate_video_value)
        if not candidate_video.is_absolute():
            candidate_video = manifest_path.parent / candidate_video
        row = CandidateRow(model, case_id, seed, candidate_video.resolve())
        _add_unique(
            rows,
            identities,
            row,
            f"{manifest_path}: line {line_number}",
        )

    return rows
