from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .case_loader import LiteCase


_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  (\S(?:.*\S)?)")
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


class CaseValidationError(RuntimeError):
    pass


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(80).startswith(_LFS_HEADER)
    except OSError as exc:
        raise CaseValidationError(f"cannot inspect {path} for Git LFS pointer: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CaseValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_bundle_file(path: Path) -> None:
    if not path.is_file():
        raise CaseValidationError(f"missing declared reference file: {path}")
    if is_lfs_pointer(path):
        raise CaseValidationError(f"Git LFS object is missing for {path}")


def _require_path_within_root(case: LiteCase, path: Path, description: str) -> None:
    try:
        path.relative_to(case.root)
    except ValueError as exc:
        raise CaseValidationError(f"{description} escapes case root: {path}") from exc


def _resolve_manifest_path(case: LiteCase, relative_path: str) -> Path:
    manifest_entry = Path(relative_path)
    if manifest_entry.is_absolute():
        raise CaseValidationError(f"checksum path escapes case root: {relative_path}")
    resolved = (case.root / manifest_entry).resolve()
    _require_path_within_root(case, resolved, "checksum path")
    return resolved


def _reference_files(case: LiteCase) -> set[Path]:
    reference_root = (case.root / "reference").resolve()
    try:
        files = {path.resolve() for path in reference_root.rglob("*") if path.is_file()}
    except OSError as exc:
        raise CaseValidationError(
            f"cannot inspect reference directory {reference_root}: {exc}"
        ) from exc
    for path in files:
        _require_path_within_root(case, path, "reference file")
    files.discard(case.checksums_path)
    return files


def _display_paths(case: LiteCase, paths: set[Path]) -> str:
    return ", ".join(sorted(str(path.relative_to(case.root)) for path in paths))


def validate_case_bundle(case: LiteCase, verify_checksums: bool = True) -> None:
    for declared_path in (
        case.reference_video,
        case.baseline_path,
        case.checksums_path,
    ):
        _require_path_within_root(case, declared_path, "declared reference path")
        _require_bundle_file(declared_path)

    try:
        lines = case.checksums_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CaseValidationError(
            f"cannot read checksum manifest {case.checksums_path}: {exc}"
        ) from exc

    entries: dict[Path, tuple[str, str]] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise CaseValidationError(
                f"invalid checksum line {line_number} in {case.checksums_path}"
            )
        expected, relative_path = match.groups()
        target = _resolve_manifest_path(case, relative_path)
        if target in entries:
            raise CaseValidationError(f"duplicate checksum path: {relative_path}")
        _require_bundle_file(target)
        entries[target] = (expected, relative_path)

    reference_files = _reference_files(case)
    entry_paths = set(entries)
    missing = reference_files - entry_paths
    extra = entry_paths - reference_files
    if missing:
        raise CaseValidationError(
            f"checksum manifest is incomplete; missing: {_display_paths(case, missing)}"
        )
    if extra:
        raise CaseValidationError(
            f"checksum manifest has unexpected paths: {_display_paths(case, extra)}"
        )

    if verify_checksums:
        for target, (expected, relative_path) in entries.items():
            actual = sha256_file(target)
            if actual != expected:
                raise CaseValidationError(
                    f"checksum mismatch for {relative_path}: expected {expected}, got {actual}"
                )
