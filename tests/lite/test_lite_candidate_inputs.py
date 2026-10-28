from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluation.context import VideoMetadata


def _write_manifest(path: Path, rows: list[Any]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _metadata(
    path: Path,
    *,
    fps: float = 10.0,
    frame_count: int = 100,
    duration_sec: float | None = 10.0,
) -> VideoMetadata:
    return VideoMetadata(
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        width=832,
        height=464,
        duration_sec=duration_sec,
    )


def _mock_successful_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation.lite import validation

    monkeypatch.setattr(
        validation.video_io,
        "load_frames_by_indices",
        lambda path, indices: [object() for _ in indices],
    )


def test_discover_candidate_directory_uses_model_root_name_and_exact_patterns(
    tmp_path: Path,
) -> None:
    from evaluation.lite.candidate_discovery import discover_candidate_directory

    model_root = tmp_path / "my-model"
    official_cases = [
        "beamdomino/canonical",
        "beamdomino/glancing",
        "beamdomino/direct_strike",
        "liquid_container/rest",
        "liquid_container/step_hold",
        "liquid_container/impulse_decay",
        "liquid_container/chirp_strafe",
    ]
    included = [
        model_root / case_id / f"seed-{index}.mp4"
        for index, case_id in enumerate(official_cases)
    ]
    excluded = [
        model_root / "beamdomino" / "unknown" / "seed-c.mp4",
        model_root / "beamdomino" / "canonical" / "nested" / "seed-d.mp4",
        model_root / "beamdomino" / "glancing" / "seed-e.mov",
    ]
    for path in [*included, *excluded]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    rows = discover_candidate_directory(model_root)

    assert [
        (row.model, row.case_id, row.seed, row.candidate_video) for row in rows
    ] == sorted(
        [
            ("my-model", case_id, path.stem, path.resolve())
            for case_id, path in zip(official_cases, included)
        ]
    )


def test_discover_candidate_directory_accepts_spaced_dashed_model_and_seed(
    tmp_path: Path,
) -> None:
    from evaluation.lite.candidate_discovery import discover_candidate_directory

    candidate = (
        tmp_path
        / "模型 name-v2"
        / "beamdomino"
        / "canonical"
        / "séed name-1.mp4"
    )
    candidate.parent.mkdir(parents=True)
    candidate.touch()

    [row] = discover_candidate_directory(candidate.parents[2])

    assert (row.model, row.seed) == ("模型 name-v2", "séed name-1")


def test_discover_candidate_directory_uses_sorted_allowed_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import candidate_discovery

    patterns: list[str] = []
    monkeypatch.setattr(
        candidate_discovery.Path,
        "glob",
        lambda path, pattern: patterns.append(pattern) or [],
    )

    candidate_discovery.discover_candidate_directory(tmp_path / "model")

    assert patterns == [
        f"{case_id}/*.mp4" for case_id in sorted(candidate_discovery.ALLOWED_CASES)
    ]


@pytest.mark.parametrize("model_root", [Path("/"), Path(".") / ".."])
def test_discover_candidate_directory_rejects_empty_or_parent_model_root_name(
    model_root: Path,
) -> None:
    from evaluation.lite.candidate_discovery import discover_candidate_directory

    with pytest.raises(
        ValueError, match="model root.*model.*(nonblank|filesystem path component)"
    ):
        discover_candidate_directory(model_root)


@pytest.mark.parametrize("seed", [".", "..", r"a\b"])
def test_discover_candidate_directory_rejects_unsafe_filename_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: str
) -> None:
    from evaluation.lite import candidate_discovery

    candidate = SimpleNamespace(stem=seed)
    monkeypatch.setattr(
        candidate_discovery.Path,
        "glob",
        lambda path, pattern: [candidate]
        if pattern == "beamdomino/canonical/*.mp4"
        else [],
    )

    with pytest.raises(ValueError, match="seed.*filesystem path component"):
        candidate_discovery.discover_candidate_directory(tmp_path / "model")


def test_discover_candidate_directory_rejects_duplicate_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import candidate_discovery

    model_root = tmp_path / "model"
    candidate = model_root / "beamdomino" / "canonical" / "seed.mp4"

    def duplicate_candidate(path: Path, pattern: str) -> list[Path]:
        if pattern == "beamdomino/canonical/*.mp4":
            return [candidate, candidate]
        return []

    monkeypatch.setattr(candidate_discovery.Path, "glob", duplicate_candidate)

    with pytest.raises(ValueError, match="duplicate candidate identity"):
        candidate_discovery.discover_candidate_directory(model_root)


def test_candidate_row_is_frozen_and_orderable() -> None:
    from evaluation.lite.candidate_discovery import CandidateRow

    later = CandidateRow("model", "liquid_container/rest", "seed", Path("b.mp4"))
    earlier = CandidateRow("model", "beamdomino/canonical", "seed", Path("a.mp4"))

    assert sorted([later, earlier]) == [earlier, later]
    with pytest.raises(FrozenInstanceError):
        earlier.seed = "changed"  # type: ignore[misc]


def test_candidate_validation_error_subclasses_runtime_error() -> None:
    from evaluation.lite.validation import CandidateValidationError

    assert issubclass(CandidateValidationError, RuntimeError)


def test_load_candidate_manifest_accepts_and_ignores_auxiliary_fields(
    tmp_path: Path,
) -> None:
    from evaluation.lite.candidate_discovery import CandidateRow, load_candidate_manifest

    manifest = tmp_path / "inputs.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "model": "model",
                "case_id": "beamdomino/canonical",
                "seed": "seed",
                "candidate_video": "candidate.mp4",
                "metadata": {"source": "submission"},
                "score": 0.75,
            }
        ],
    )

    assert load_candidate_manifest(manifest) == [
        CandidateRow(
            model="model",
            case_id="beamdomino/canonical",
            seed="seed",
            candidate_video=(tmp_path / "candidate.mp4").resolve(),
        )
    ]


def test_load_candidate_manifest_accepts_spaced_dashed_identity(tmp_path: Path) -> None:
    from evaluation.lite.candidate_discovery import CandidateRow, load_candidate_manifest

    manifest = tmp_path / "inputs.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "model": "模型 name-v2",
                "case_id": "beamdomino/canonical",
                "seed": "séed name-1",
                "candidate_video": "candidate.mp4",
            }
        ],
    )

    assert load_candidate_manifest(manifest) == [
        CandidateRow(
            model="模型 name-v2",
            case_id="beamdomino/canonical",
            seed="séed name-1",
            candidate_video=(tmp_path / "candidate.mp4").resolve(),
        )
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "/abs"),
        ("model", "../escape"),
        ("model", "a/b"),
        ("model", r"a\b"),
        ("model", "."),
        ("model", ".."),
        ("seed", "/abs"),
        ("seed", "../escape"),
        ("seed", "a/b"),
        ("seed", r"a\b"),
        ("seed", "."),
        ("seed", ".."),
        ("seed", "nul\0value"),
    ],
)
def test_load_candidate_manifest_rejects_unsafe_identity_components(
    tmp_path: Path, field: str, value: str
) -> None:
    from evaluation.lite.candidate_discovery import load_candidate_manifest

    row = {
        "model": "model",
        "case_id": "beamdomino/canonical",
        "seed": "seed",
        "candidate_video": "candidate.mp4",
    }
    row[field] = value
    manifest = tmp_path / "inputs.jsonl"
    _write_manifest(manifest, [row])

    with pytest.raises(
        ValueError, match=rf"line 1.*{field}.*filesystem path component"
    ):
        load_candidate_manifest(manifest)


@pytest.mark.parametrize("duplicate_field", ["model", "metadata"])
def test_load_candidate_manifest_rejects_duplicate_json_object_keys(
    tmp_path: Path, duplicate_field: str
) -> None:
    from evaluation.lite.candidate_discovery import load_candidate_manifest

    manifest = tmp_path / "inputs.jsonl"
    first_value = '"model"' if duplicate_field == "model" else '{"source":"first"}'
    manifest.write_text(
        '{"model":"model","case_id":"beamdomino/canonical","seed":"seed",'
        f'"candidate_video":"candidate.mp4","{duplicate_field}":{first_value},'
        f'"{duplicate_field}":"duplicate"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match=rf"line 1.*duplicate JSON object key.*{duplicate_field}"
    ):
        load_candidate_manifest(manifest)


def test_load_candidate_manifest_rejects_duplicate_nested_auxiliary_key(
    tmp_path: Path,
) -> None:
    from evaluation.lite.candidate_discovery import load_candidate_manifest

    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        '{"model":"model","case_id":"beamdomino/canonical","seed":"seed",'
        '"candidate_video":"candidate.mp4",'
        '"metadata":{"source":"first","source":"duplicate"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match=r"line 1.*duplicate JSON object key.*source"
    ):
        load_candidate_manifest(manifest)


def test_discover_candidate_directory_keeps_distinct_seed_identity_for_alias(
    tmp_path: Path,
) -> None:
    from evaluation.lite.candidate_discovery import discover_candidate_directory

    model_root = tmp_path / "model"
    target = model_root / "beamdomino" / "canonical" / "seed.mp4"
    target.parent.mkdir(parents=True)
    target.touch()
    alias = target.parent / "seed-alias.mp4"
    alias.symlink_to(target.name)

    rows = discover_candidate_directory(model_root)

    assert [row.seed for row in rows] == ["seed", "seed-alias"]


def test_load_candidate_manifest_resolves_relative_paths_and_preserves_order(
    tmp_path: Path,
) -> None:
    from evaluation.lite.candidate_discovery import load_candidate_manifest

    manifest = tmp_path / "inputs.jsonl"
    absolute = tmp_path / "absolute.mp4"
    _write_manifest(
        manifest,
        [
            {
                "model": "z-model",
                "case_id": "liquid_container/rest",
                "seed": "second",
                "candidate_video": "videos/second.mp4",
            },
            {
                "model": "a-model",
                "case_id": "beamdomino/canonical",
                "seed": "first",
                "candidate_video": str(absolute),
            },
        ],
    )

    rows = load_candidate_manifest(manifest)

    assert [row.seed for row in rows] == ["second", "first"]
    assert rows[0].candidate_video == (tmp_path / "videos" / "second.mp4").resolve()
    assert rows[1].candidate_video == absolute.resolve()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{bad json", "line 1.*invalid JSON"),
        ("[]", "line 1.*JSON object"),
        ('{"model": "m"}', "line 1.*missing required field.*case_id"),
    ],
)
def test_load_candidate_manifest_reports_malformed_rows(
    tmp_path: Path, line: str, message: str
) -> None:
    from evaluation.lite.candidate_discovery import load_candidate_manifest

    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(f"\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message.replace("line 1", "line 2")):
        load_candidate_manifest(manifest)


@pytest.mark.parametrize("field", ["model", "case_id", "seed", "candidate_video"])
@pytest.mark.parametrize("value", ["", " ", 1, True, None])
def test_load_candidate_manifest_rejects_blank_or_non_string_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    from evaluation.lite.candidate_discovery import load_candidate_manifest

    row = {
        "model": "model",
        "case_id": "beamdomino/canonical",
        "seed": "seed",
        "candidate_video": "candidate.mp4",
    }
    row[field] = value
    manifest = tmp_path / "inputs.jsonl"
    _write_manifest(manifest, [row])

    with pytest.raises(ValueError, match=rf"line 1.*{field}.*nonblank string"):
        load_candidate_manifest(manifest)


def test_load_candidate_manifest_rejects_non_official_case(tmp_path: Path) -> None:
    from evaluation.lite.candidate_discovery import load_candidate_manifest

    manifest = tmp_path / "inputs.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "model": "model",
                "case_id": "beamdomino/unknown",
                "seed": "seed",
                "candidate_video": "candidate.mp4",
            }
        ],
    )

    with pytest.raises(ValueError, match="line 1.*unsupported official Lite case_id"):
        load_candidate_manifest(manifest)


def test_load_candidate_manifest_rejects_duplicate_identity(tmp_path: Path) -> None:
    from evaluation.lite.candidate_discovery import load_candidate_manifest

    row = {
        "model": "model",
        "case_id": "beamdomino/canonical",
        "seed": "seed",
        "candidate_video": "candidate.mp4",
    }
    manifest = tmp_path / "inputs.jsonl"
    _write_manifest(manifest, [row, {**row, "candidate_video": "other.mp4"}])

    with pytest.raises(ValueError, match="line 2.*duplicate candidate identity"):
        load_candidate_manifest(manifest)


@pytest.mark.parametrize(
    ("candidate_duration", "reference_duration", "expected"),
    [(4.9, 10.0, True), (5.0, 10.0, False), (0.9, 1.5, True)],
)
def test_is_severely_truncated(
    candidate_duration: float, reference_duration: float, expected: bool
) -> None:
    from evaluation.lite.validation import is_severely_truncated

    assert (
        is_severely_truncated(
            candidate_duration=candidate_duration,
            reference_duration=reference_duration,
        )
        is expected
    )


def test_validate_candidate_video_returns_candidate_and_reference_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import validation

    candidate = tmp_path / "candidate.mp4"
    reference = tmp_path / "reference.mp4"
    candidate_metadata = _metadata(candidate)
    reference_metadata = _metadata(reference)
    metadata = {
        str(candidate): candidate_metadata,
        str(reference): reference_metadata,
    }
    monkeypatch.setattr(
        validation.video_io, "get_video_metadata", lambda path: metadata[path]
    )
    calls: list[tuple[str, list[int]]] = []
    monkeypatch.setattr(
        validation.video_io,
        "load_frames_by_indices",
        lambda path, indices: calls.append((path, list(indices)))
        or [object() for _ in indices],
    )

    result = validation.validate_candidate_video(candidate, reference)

    assert result.candidate == candidate_metadata
    assert result.reference == reference_metadata
    assert calls == [
        (str(candidate), [0, 99]),
        (str(reference), [0, 99]),
    ]


@pytest.mark.parametrize("frame_count", [1, 2])
def test_validate_candidate_video_decodes_unique_first_and_last_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame_count: int
) -> None:
    from evaluation.lite import validation

    calls: list[list[int]] = []
    monkeypatch.setattr(
        validation.video_io,
        "get_video_metadata",
        lambda path: _metadata(Path(path), frame_count=frame_count),
    )
    monkeypatch.setattr(
        validation.video_io,
        "load_frames_by_indices",
        lambda path, indices: calls.append(list(indices))
        or [object() for _ in indices],
    )

    validation.validate_candidate_video(
        tmp_path / "candidate.mp4", tmp_path / "reference.mp4"
    )

    expected = [0] if frame_count == 1 else [0, 1]
    assert calls == [expected, expected]


@pytest.mark.parametrize("failed_role", ["candidate", "reference"])
def test_validate_candidate_video_rejects_undecodable_video_with_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_role: str
) -> None:
    from evaluation.lite import validation

    candidate = tmp_path / "candidate.mp4"
    reference = tmp_path / "reference.mp4"
    failed_path = candidate if failed_role == "candidate" else reference
    monkeypatch.setattr(
        validation.video_io, "get_video_metadata", lambda path: _metadata(Path(path))
    )
    monkeypatch.setattr(
        validation.video_io,
        "load_frames_by_indices",
        lambda path, indices: []
        if path == str(failed_path)
        else [object() for _ in indices],
    )

    with pytest.raises(
        validation.CandidateValidationError,
        match=rf"{failed_role}.*{failed_path}.*could not decode any frames",
    ):
        validation.validate_candidate_video(candidate, reference)


def test_validate_candidate_video_accepts_overestimated_frame_count_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A healthy clip whose reported last index overshoots the real stream
    (a common OpenCV CAP_PROP_FRAME_COUNT over-estimate) must not be rejected:
    decoding the first frame succeeds even though the tail index does not."""
    from evaluation.lite import validation

    candidate = tmp_path / "candidate.mp4"
    reference = tmp_path / "reference.mp4"
    monkeypatch.setattr(
        validation.video_io, "get_video_metadata", lambda path: _metadata(Path(path))
    )
    # The real loader stops at the first unreadable index, so requesting
    # [0, last] returns only the first frame when the tail is over-reported.
    monkeypatch.setattr(
        validation.video_io,
        "load_frames_by_indices",
        lambda path, indices: [object()],
    )

    result = validation.validate_candidate_video(candidate, reference)

    assert result.candidate.frame_count == 100


@pytest.mark.parametrize("failed_role", ["candidate", "reference"])
def test_validate_candidate_video_wraps_decode_error_with_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_role: str
) -> None:
    from evaluation.lite import validation

    candidate = tmp_path / "candidate.mp4"
    reference = tmp_path / "reference.mp4"
    failed_path = candidate if failed_role == "candidate" else reference
    monkeypatch.setattr(
        validation.video_io, "get_video_metadata", lambda path: _metadata(Path(path))
    )

    def decode(path: str, indices: list[int]) -> list[object]:
        if path == str(failed_path):
            raise RuntimeError("corrupt frame data")
        return [object() for _ in indices]

    monkeypatch.setattr(validation.video_io, "load_frames_by_indices", decode)

    with pytest.raises(
        validation.CandidateValidationError,
        match=rf"cannot decode {failed_role} video {failed_path}: corrupt frame data",
    ):
        validation.validate_candidate_video(candidate, reference)


def test_validate_candidate_video_wraps_undecodable_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import validation

    monkeypatch.setattr(
        validation.video_io,
        "get_video_metadata",
        lambda path: (_ for _ in ()).throw(RuntimeError("Failed to open video")),
    )

    with pytest.raises(
        validation.CandidateValidationError, match="candidate.*Failed to open"
    ):
        validation.validate_candidate_video(
            tmp_path / "candidate.mp4", tmp_path / "reference.mp4"
        )


@pytest.mark.parametrize(
    ("failed_role", "probe_error"),
    [
        ("candidate", FileNotFoundError("missing video")),
        ("reference", PermissionError("permission denied")),
    ],
)
def test_validate_candidate_video_wraps_missing_or_unreadable_video_with_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_role: str,
    probe_error: OSError,
) -> None:
    from evaluation.lite import validation

    candidate = tmp_path / "missing-candidate.mp4"
    reference = tmp_path / "unreadable-reference.mp4"
    failed_path = candidate if failed_role == "candidate" else reference

    def probe(path: str) -> VideoMetadata:
        if path == str(failed_path):
            raise probe_error
        return _metadata(Path(path))

    monkeypatch.setattr(validation.video_io, "get_video_metadata", probe)

    with pytest.raises(
        validation.CandidateValidationError,
        match=rf"cannot probe {failed_role} video {failed_path}: {probe_error}",
    ):
        validation.validate_candidate_video(candidate, reference)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frame_count": 0}, "zero frames"),
        ({"fps": 0.0}, "nonpositive FPS"),
        ({"duration_sec": None}, "invalid duration"),
        ({"duration_sec": True}, "invalid duration"),
        ({"duration_sec": 0.0}, "nonpositive duration"),
        ({"duration_sec": float("nan")}, "invalid duration"),
        ({"duration_sec": float("inf")}, "invalid duration"),
        ({"duration_sec": float("-inf")}, "invalid duration"),
    ],
)
def test_validate_candidate_video_rejects_invalid_candidate_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    from evaluation.lite import validation

    candidate = tmp_path / "candidate.mp4"
    reference = tmp_path / "reference.mp4"
    candidate_metadata = _metadata(candidate, **overrides)  # type: ignore[arg-type]
    reference_metadata = _metadata(reference)
    monkeypatch.setattr(
        validation.video_io,
        "get_video_metadata",
        lambda path: candidate_metadata
        if path == str(candidate)
        else reference_metadata,
    )
    _mock_successful_decoding(monkeypatch)

    with pytest.raises(validation.CandidateValidationError, match=message):
        validation.validate_candidate_video(candidate, reference)


def test_validate_candidate_video_rejects_invalid_reference_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import validation

    candidate = tmp_path / "candidate.mp4"
    reference = tmp_path / "reference.mp4"
    monkeypatch.setattr(
        validation.video_io,
        "get_video_metadata",
        lambda path: _metadata(candidate)
        if path == str(candidate)
        else _metadata(reference, fps=-1.0),
    )
    _mock_successful_decoding(monkeypatch)

    with pytest.raises(
        validation.CandidateValidationError, match="reference.*nonpositive FPS"
    ):
        validation.validate_candidate_video(candidate, reference)


def test_validate_candidate_video_rejects_severe_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import validation

    candidate = tmp_path / "candidate.mp4"
    reference = tmp_path / "reference.mp4"
    monkeypatch.setattr(
        validation.video_io,
        "get_video_metadata",
        lambda path: _metadata(candidate, duration_sec=4.9)
        if path == str(candidate)
        else _metadata(reference, duration_sec=10.0),
    )
    _mock_successful_decoding(monkeypatch)

    with pytest.raises(validation.CandidateValidationError, match="severely truncated"):
        validation.validate_candidate_video(candidate, reference)


def test_normalize_liquid_video_uses_locked_ffmpeg_command_and_creates_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import normalization

    source = tmp_path / "source.mp4"
    destination = tmp_path / "nested" / "normalized.mp4"
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(
        normalization.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    normalization.normalize_liquid_video(source, destination)

    assert destination.parent.is_dir()
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        "scale=832:464:flags=bilinear,fps=16",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == normalization.DEFAULT_NORMALIZATION_TIMEOUT_SEC


def test_normalize_liquid_video_surfaces_ffmpeg_stderr_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import normalization

    def fail(command: list[str], **kwargs: Any) -> None:
        raise normalization.subprocess.CalledProcessError(
            returncode=1, cmd=command, stderr="moov atom not found"
        )

    monkeypatch.setattr(normalization.subprocess, "run", fail)

    with pytest.raises(
        normalization.NormalizationError, match="exit 1.*moov atom not found"
    ):
        normalization.normalize_liquid_video(
            tmp_path / "in.mp4", tmp_path / "out.mp4"
        )


def test_normalize_liquid_video_reports_missing_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import normalization

    def missing(command: list[str], **kwargs: Any) -> None:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(normalization.subprocess, "run", missing)

    with pytest.raises(normalization.NormalizationError, match="ffmpeg executable not found"):
        normalization.normalize_liquid_video(
            tmp_path / "in.mp4", tmp_path / "out.mp4"
        )
