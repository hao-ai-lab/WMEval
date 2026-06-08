from __future__ import annotations

import json
from pathlib import Path
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


def test_discover_candidate_directory_uses_model_root_name_and_exact_patterns(
    tmp_path: Path,
) -> None:
    from evaluation.lite.candidate_discovery import discover_candidate_directory

    model_root = tmp_path / "my-model"
    included = [
        model_root / "liquid_container" / "rest" / "seed-b.mp4",
        model_root / "beamdomino" / "canonical" / "seed-a.mp4",
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
    ] == [
        ("my-model", "beamdomino/canonical", "seed-a", included[1].resolve()),
        ("my-model", "liquid_container/rest", "seed-b", included[0].resolve()),
    ]


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
        (
            '{"model": "m", "case_id": "beamdomino/canonical", "seed": "s", '
            '"candidate_video": "v.mp4", "extra": true}',
            "line 1.*unexpected field.*extra",
        ),
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

    result = validation.validate_candidate_video(candidate, reference)

    assert result.candidate == candidate_metadata
    assert result.reference == reference_metadata


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
    ("overrides", "message"),
    [
        ({"frame_count": 0}, "zero frames"),
        ({"fps": 0.0}, "nonpositive FPS"),
        ({"duration_sec": None}, "invalid duration"),
        ({"duration_sec": True}, "invalid duration"),
        ({"duration_sec": 0.0}, "nonpositive duration"),
        ({"duration_sec": float("nan")}, "invalid duration"),
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

    with pytest.raises(validation.CandidateValidationError, match="severely truncated"):
        validation.validate_candidate_video(candidate, reference)


def test_normalize_liquid_video_uses_locked_ffmpeg_command_and_creates_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import normalization

    source = tmp_path / "source.mp4"
    destination = tmp_path / "nested" / "normalized.mp4"
    calls: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(
        normalization.subprocess,
        "run",
        lambda command, check: calls.append((command, check)),
    )

    normalization.normalize_liquid_video(source, destination)

    assert destination.parent.is_dir()
    assert calls == [
        (
            [
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
            ],
            True,
        )
    ]
