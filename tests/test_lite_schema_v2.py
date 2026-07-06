"""Tests for the schema_version 2 / suite-parameterization adaptations of
``evaluation.lite`` (reference-free cases, ``allowed_cases`` injection,
candidate ``params`` passthrough)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from evaluation.context import VideoMetadata
from evaluation.lite.case_loader import load_case
from evaluation.lite.validation import CaseValidationError, validate_case_bundle


def _write_checksums(case_dir: Path, relative_paths: list[str]) -> None:
    lines = []
    for relative_path in relative_paths:
        digest = hashlib.sha256((case_dir / relative_path).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative_path}")
    (case_dir / "reference" / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_v2_case(
    case_dir: Path,
    *,
    case_id: str = "racing_car/first",
    scene: str = "racing_car",
    modules: list[str] | None = None,
    with_video: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    reference = case_dir / "reference"
    reference.mkdir(parents=True)
    (reference / "baseline.json").write_text("{}\n", encoding="utf-8")
    (reference / "ref.png").write_text("tiny first frame\n", encoding="utf-8")
    checksum_paths = ["reference/baseline.json", "reference/ref.png"]
    reference_block: dict[str, Any] = {
        "first_frame": "reference/ref.png",
        "baseline": "reference/baseline.json",
        "checksums": "reference/checksums.sha256",
    }
    if with_video:
        (reference / "reference.mp4").write_text("tiny reference\n", encoding="utf-8")
        checksum_paths.insert(0, "reference/reference.mp4")
        reference_block["video"] = "reference/reference.mp4"
    _write_checksums(case_dir, checksum_paths)
    raw: dict[str, Any] = {
        "schema_version": 2,
        "case_id": case_id,
        "scene": scene,
        "display_name": "Racing Car (first person)",
        "prompt": "A racing car drives around a track.",
        "perspective": "first_person",
        "rubric_scene_name": "RacingCar_first",
        "modules": modules if modules is not None else [],
        "reference": reference_block,
        "module_params": {
            "shadow_consistency": {
                "shadow_text_prompt": "car.",
                "shadow_anchor_label": "car",
            }
        },
        "protocol": {
            "candidate_normalization": "none",
            "min_duration_sec": 1.0,
            "min_duration_ratio": 0.5,
            "duration_threshold_mode": "max",
        },
        "evaluator_lock": {
            "release_id": "wmeval-lite-v1",
            "code_revision": "test",
            "evaluator_fingerprint": "test-fingerprint",
            "module_config": {},
            "artifact_schema_versions": {},
            "dependency_versions": {},
        },
    }
    if extra:
        raw.update(extra)
    (case_dir / "case.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )


def test_load_v2_reference_free_case_populates_new_fields(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir)

    case = load_case(case_dir)

    assert case.schema_version == 2
    assert case.case_id == "racing_car/first"
    assert case.reference_video is None
    assert case.first_frame == (case_dir / "reference" / "ref.png").resolve()
    assert case.prompt == "A racing car drives around a track."
    assert case.perspective == "first_person"
    assert case.rubric_scene_name == "RacingCar_first"
    assert case.modules == ()
    assert case.module_params["shadow_consistency"]["shadow_anchor_label"] == "car"


def test_load_v2_case_with_reference_video(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir, with_video=True)

    case = load_case(case_dir)

    assert case.reference_video == (case_dir / "reference" / "reference.mp4").resolve()


def test_load_v2_case_skips_membership_check_by_default(tmp_path: Path) -> None:
    case_dir = tmp_path / "brand_new_scene" / "variant"
    _write_v2_case(case_dir, case_id="brand_new_scene/variant", scene="brand_new_scene")

    case = load_case(case_dir)

    assert case.case_id == "brand_new_scene/variant"


def test_load_v2_case_enforces_injected_allowed_cases(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir)

    with pytest.raises(ValueError, match="unsupported official Lite case_id"):
        load_case(case_dir, allowed_cases={"beamdomino/canonical"})

    assert load_case(case_dir, allowed_cases={"racing_car/first"}).scene == "racing_car"


def test_load_v2_case_enforces_injected_expected_modules(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir, modules=["shadow_consistency"])

    loaded = load_case(
        case_dir, expected_modules={"racing_car": ("shadow_consistency",)}
    )
    assert loaded.modules == ("shadow_consistency",)

    with pytest.raises(ValueError, match="modules must be"):
        load_case(case_dir, expected_modules={"racing_car": ("lpips",)})

    with pytest.raises(ValueError, match="no expected modules for scene"):
        load_case(case_dir, expected_modules={"other_scene": ("lpips",)})


def test_load_v2_case_still_enforces_scene_prefix(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir, scene="wrong_scene")

    with pytest.raises(ValueError, match="does not match scene"):
        load_case(case_dir)


def test_load_v2_case_rejects_blank_optional_fields(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir, extra={"prompt": "  "})

    with pytest.raises(ValueError, match="prompt must be a nonblank string"):
        load_case(case_dir)


def test_load_v2_case_rejects_non_mapping_module_params(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir, extra={"module_params": {"shadow_consistency": ["x"]}})

    with pytest.raises(ValueError, match="module_params.shadow_consistency"):
        load_case(case_dir)


def test_v1_case_ignores_v2_only_reference_keys(tmp_path: Path) -> None:
    # A v1 case must still require reference.video even when v2-style keys
    # appear; the v2 optional fields stay at their defaults.
    case_dir = tmp_path / "beamdomino" / "canonical"
    reference = case_dir / "reference"
    reference.mkdir(parents=True)
    (reference / "reference.mp4").write_text("tiny reference\n", encoding="utf-8")
    (reference / "baseline.json").write_text("{}\n", encoding="utf-8")
    _write_checksums(
        case_dir, ["reference/reference.mp4", "reference/baseline.json"]
    )
    (case_dir / "case.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "case_id": "beamdomino/canonical",
                "scene": "beamdomino",
                "display_name": "Test Case",
                "prompt": "ignored for v1",
                "modules": ["causal_step"],
                "reference": {
                    "video": "reference/reference.mp4",
                    "baseline": "reference/baseline.json",
                    "checksums": "reference/checksums.sha256",
                },
                "protocol": {
                    "candidate_normalization": "beamdomino_native",
                    "min_duration_sec": 1.0,
                    "min_duration_ratio": 0.5,
                    "duration_threshold_mode": "max",
                },
                "evaluator_lock": {
                    "release_id": "wmeval-lite-v1",
                    "code_revision": "test",
                    "evaluator_fingerprint": "test-fingerprint",
                    "module_config": {},
                    "artifact_schema_versions": {},
                    "dependency_versions": {},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    case = load_case(case_dir)

    assert case.schema_version == 1
    assert case.prompt is None
    assert case.first_frame is None
    assert case.module_params == {}


def test_load_v2_case_resolves_reference_artifacts(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir)
    artifacts_dir = case_dir / "reference" / "artifacts" / "action_path"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "action.npy").write_bytes(b"fake")

    def add_artifacts(raw: dict[str, Any]) -> None:
        raw["reference"]["artifacts"] = {
            "synthetic_action_dir": "reference/artifacts/action_path",
        }

    raw = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
    add_artifacts(raw)
    (case_dir / "case.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )

    case = load_case(case_dir)

    assert case.reference_artifacts["synthetic_action_dir"] == artifacts_dir.resolve()


def test_load_v2_case_rejects_blank_reference_artifact_path(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir)
    raw = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))
    raw["reference"]["artifacts"] = {"qa_dataset": "  "}
    (case_dir / "case.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="reference.artifacts.qa_dataset"):
        load_case(case_dir)


def test_validate_case_bundle_accepts_reference_free_v2(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir)

    validate_case_bundle(load_case(case_dir))


def test_validate_case_bundle_rejects_missing_first_frame(tmp_path: Path) -> None:
    case_dir = tmp_path / "racing_car" / "first"
    _write_v2_case(case_dir)
    (case_dir / "reference" / "ref.png").unlink()

    with pytest.raises(CaseValidationError, match="missing declared reference file"):
        validate_case_bundle(load_case(case_dir))


def test_discover_candidate_directory_accepts_injected_allowed_cases(
    tmp_path: Path,
) -> None:
    from evaluation.lite.candidate_discovery import discover_candidate_directory

    model_root = tmp_path / "genie3"
    (model_root / "racing_car" / "first").mkdir(parents=True)
    (model_root / "racing_car" / "first" / "0001.mp4").write_bytes(b"")

    assert discover_candidate_directory(model_root) == []

    rows = discover_candidate_directory(
        model_root, allowed_cases={"racing_car/first"}
    )
    assert [(row.case_id, row.seed) for row in rows] == [("racing_car/first", "0001")]


def test_load_candidate_manifest_accepts_injected_allowed_cases_and_params(
    tmp_path: Path,
) -> None:
    from evaluation.lite.candidate_discovery import load_candidate_manifest

    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "model": "genie3",
                "case_id": "racing_car/first",
                "seed": "0001",
                "candidate_video": "candidate.mp4",
                "params": {"shadow_hud_preset": "genie"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported official Lite case_id"):
        load_candidate_manifest(manifest)

    rows = load_candidate_manifest(manifest, allowed_cases={"racing_car/first"})
    assert len(rows) == 1
    assert rows[0].params == {"shadow_hud_preset": "genie"}


def test_load_candidate_manifest_rejects_non_object_params(tmp_path: Path) -> None:
    from evaluation.lite.candidate_discovery import load_candidate_manifest

    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "model": "genie3",
                "case_id": "beamdomino/canonical",
                "seed": "0001",
                "candidate_video": "candidate.mp4",
                "params": ["not", "a", "mapping"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="params must be a JSON object"):
        load_candidate_manifest(manifest)


def test_candidate_row_params_excluded_from_identity() -> None:
    from evaluation.lite.candidate_discovery import CandidateRow

    plain = CandidateRow("model", "beamdomino/canonical", "seed", Path("a.mp4"))
    with_params = CandidateRow(
        "model",
        "beamdomino/canonical",
        "seed",
        Path("a.mp4"),
        {"shadow_hud_preset": "genie"},
    )

    assert plain == with_params
    assert sorted([with_params, plain]) == [with_params, plain]


def _metadata(path: Path, *, duration_sec: float) -> VideoMetadata:
    return VideoMetadata(
        path=str(path),
        fps=10.0,
        frame_count=int(duration_sec * 10),
        width=832,
        height=464,
        duration_sec=duration_sec,
    )


def test_validate_candidate_video_reference_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import validation

    candidate = tmp_path / "candidate.mp4"
    monkeypatch.setattr(
        validation.video_io,
        "get_video_metadata",
        lambda path: _metadata(candidate, duration_sec=5.0),
    )
    monkeypatch.setattr(
        validation.video_io,
        "load_frames_by_indices",
        lambda path, indices: [object() for _ in indices],
    )

    result = validation.validate_candidate_video(candidate)

    assert result.reference is None
    assert result.candidate.duration_sec == 5.0


def test_validate_candidate_video_reference_free_rejects_below_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite import validation

    candidate = tmp_path / "candidate.mp4"
    monkeypatch.setattr(
        validation.video_io,
        "get_video_metadata",
        lambda path: _metadata(candidate, duration_sec=0.4),
    )
    monkeypatch.setattr(
        validation.video_io,
        "load_frames_by_indices",
        lambda path, indices: [object() for _ in indices],
    )

    with pytest.raises(
        validation.CandidateValidationError, match="severely truncated"
    ):
        validation.validate_candidate_video(candidate, min_duration_sec=1.0)
