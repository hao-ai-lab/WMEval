from __future__ import annotations

import hashlib
import math
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

from evaluation import video_io
from evaluation.context import VideoMetadata

from .case_loader import LiteCase


_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  (\S(?:.*\S)?)")
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"

# Locked v1 severe-truncation defaults, mirrored in every case.yaml protocol as
# `min_duration_sec` / `min_duration_ratio`. Kept here as the single source for
# the function defaults; the execution layer should pass each case's protocol
# values explicitly rather than relying on these.
DEFAULT_MIN_DURATION_SEC = 1.0
DEFAULT_MIN_DURATION_RATIO = 0.5


class CaseValidationError(RuntimeError):
    pass


class CandidateValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateVideoMetadata:
    candidate: VideoMetadata
    # None for reference-free dimensions (no reference video to compare with).
    reference: VideoMetadata | None


def is_severely_truncated(
    *,
    candidate_duration: float,
    reference_duration: float,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    min_duration_ratio: float = DEFAULT_MIN_DURATION_RATIO,
) -> bool:
    threshold = max(min_duration_sec, reference_duration * min_duration_ratio)
    return candidate_duration < threshold


def _probe_video(
    path: str | Path, role: str, *, fps_override: float | None = None
) -> VideoMetadata:
    try:
        # Only pass the override when set: keeps the historical call shape
        # (and every test double that mimics it) for the common case.
        if fps_override is None:
            metadata = video_io.get_video_metadata(str(path))
        else:
            metadata = video_io.get_video_metadata(str(path), fps_override=fps_override)
    except Exception as exc:
        raise CandidateValidationError(
            f"cannot probe {role} video {path}: {exc}"
        ) from exc

    if metadata.frame_count <= 0:
        raise CandidateValidationError(f"{role} video has zero frames: {path}")
    if not math.isfinite(metadata.fps) or metadata.fps <= 0:
        raise CandidateValidationError(f"{role} video has nonpositive FPS: {path}")
    duration = metadata.duration_sec
    if (
        duration is None
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
    ):
        raise CandidateValidationError(f"{role} video has invalid duration: {path}")
    if duration <= 0:
        raise CandidateValidationError(f"{role} video has nonpositive duration: {path}")
    return metadata


def _verify_decodable(path: str | Path, role: str, metadata: VideoMetadata) -> None:
    """Confirm the stream actually yields a decoded frame.

    OpenCV's ``CAP_PROP_FRAME_COUNT`` is frequently an over-estimate (it is
    derived from duration x fps for many containers/codecs), so seeking to the
    reported final index can fail on a perfectly healthy clip and previously
    triggered a false "cannot decode" rejection. We probe the first and
    reported-last indices but only require that *some* frame decodes: a healthy
    clip whose tail index overshoots the real stream is accepted, while a
    corrupt or empty clip (no frame decodes, or decoding raises) is rejected.
    Severe truncation is handled separately by the duration check.
    """
    indices = sorted({0, max(metadata.frame_count - 1, 0)})
    try:
        frames = video_io.load_frames_by_indices(str(path), indices)
    except Exception as exc:
        raise CandidateValidationError(
            f"cannot decode {role} video {path}: {exc}"
        ) from exc
    if not frames:
        raise CandidateValidationError(
            f"{role} video {path} could not decode any frames "
            f"(reported {metadata.frame_count} frames)"
        )


def validate_candidate_video(
    candidate_video: str | Path,
    reference_video: str | Path | None = None,
    *,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    min_duration_ratio: float = DEFAULT_MIN_DURATION_RATIO,
    candidate_fps_override: float | None = None,
) -> CandidateVideoMetadata:
    """Validate a candidate video, optionally against a reference video.

    ``reference_video=None`` (reference-free dimensions) still checks that the
    candidate decodes and meets the absolute ``min_duration_sec`` floor; the
    reference-relative ratio check is skipped.

    ``candidate_fps_override``, when given, is used in place of the
    candidate's own container-reported fps for both this duration check and
    (via the same value passed to the runner) all downstream frame-time
    alignment -- see ``evaluation.video_io.get_video_metadata``.
    """
    candidate = _probe_video(candidate_video, "candidate", fps_override=candidate_fps_override)

    if reference_video is None:
        _verify_decodable(candidate_video, "candidate", candidate)
        assert candidate.duration_sec is not None
        if candidate.duration_sec < min_duration_sec:
            raise CandidateValidationError(
                f"candidate video is severely truncated: {candidate_video} "
                f"({candidate.duration_sec}s candidate, "
                f"{min_duration_sec}s minimum)"
            )
        return CandidateVideoMetadata(candidate=candidate, reference=None)

    reference = _probe_video(reference_video, "reference")
    _verify_decodable(candidate_video, "candidate", candidate)
    _verify_decodable(reference_video, "reference", reference)
    assert candidate.duration_sec is not None
    assert reference.duration_sec is not None
    if is_severely_truncated(
        candidate_duration=candidate.duration_sec,
        reference_duration=reference.duration_sec,
        min_duration_sec=min_duration_sec,
        min_duration_ratio=min_duration_ratio,
    ):
        raise CandidateValidationError(
            f"candidate video is severely truncated: {candidate_video} "
            f"({candidate.duration_sec}s candidate, "
            f"{reference.duration_sec}s reference)"
        )
    return CandidateVideoMetadata(candidate=candidate, reference=reference)


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


def _canonical_checksum_path(relative_path: str) -> str:
    if (
        PurePosixPath(relative_path).is_absolute()
        or "\\" in relative_path
        or posixpath.normpath(relative_path) != relative_path
    ):
        raise CaseValidationError(f"noncanonical checksum path: {relative_path}")
    return relative_path


def _resolve_manifest_path(case: LiteCase, relative_path: str) -> Path:
    if PurePosixPath(relative_path).is_absolute():
        raise CaseValidationError(f"checksum path escapes case root: {relative_path}")
    resolved = (case.root / Path(relative_path)).resolve()
    _require_path_within_root(case, resolved, "checksum path")
    return resolved


def _reference_files(case: LiteCase) -> dict[str, Path]:
    reference_root = case.root / "reference"
    try:
        paths = [reference_root, *reference_root.rglob("*")]
        files: dict[str, Path] = {}
        for path in paths:
            if path.is_symlink():
                raise CaseValidationError(f"symlink is not allowed under reference: {path}")
            if not path.is_file():
                continue
            resolved = path.resolve()
            _require_path_within_root(case, resolved, "reference file")
            lexical_name = path.relative_to(case.root).as_posix()
            files[lexical_name] = resolved
    except OSError as exc:
        raise CaseValidationError(
            f"cannot inspect reference directory {reference_root}: {exc}"
        ) from exc
    files.pop(case.checksums_path.relative_to(case.root).as_posix(), None)
    return files


def _display_paths(paths: set[str]) -> str:
    return ", ".join(sorted(paths))


def validate_case_bundle(case: LiteCase, verify_checksums: bool = True) -> None:
    declared_paths = [
        path
        for path in (
            case.reference_video,
            case.first_frame,
            case.baseline_path,
            case.checksums_path,
        )
        if path is not None
    ]
    for declared_path in declared_paths:
        _require_path_within_root(case, declared_path, "declared reference path")
        _require_bundle_file(declared_path)

    try:
        lines = case.checksums_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CaseValidationError(
            f"cannot read checksum manifest {case.checksums_path}: {exc}"
        ) from exc

    entries: dict[str, tuple[str, Path]] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise CaseValidationError(
                f"invalid checksum line {line_number} in {case.checksums_path}"
            )
        expected, relative_path = match.groups()
        lexical_name = _canonical_checksum_path(relative_path)
        target = _resolve_manifest_path(case, lexical_name)
        if lexical_name in entries:
            raise CaseValidationError(f"duplicate checksum path: {relative_path}")
        _require_bundle_file(target)
        entries[lexical_name] = (expected, target)

    reference_files = _reference_files(case)
    entry_names = set(entries)
    reference_names = set(reference_files)
    missing = reference_names - entry_names
    extra = entry_names - reference_names
    if missing:
        raise CaseValidationError(
            f"checksum manifest is incomplete; missing: {_display_paths(missing)}"
        )
    if extra:
        raise CaseValidationError(
            f"checksum manifest has unexpected paths: {_display_paths(extra)}"
        )

    if verify_checksums:
        for relative_path, (expected, target) in entries.items():
            actual = sha256_file(target)
            if actual != expected:
                raise CaseValidationError(
                    f"checksum mismatch for {relative_path}: expected {expected}, got {actual}"
                )
