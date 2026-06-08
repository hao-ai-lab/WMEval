from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from evaluation.lite.case_loader import (
    ALLOWED_CASES,
    SCENE_MODULES,
    discover_cases,
    load_case,
)
from evaluation.lite.validation import CaseValidationError, validate_case_bundle


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "lite"


def _write_case(
    case_dir: Path,
    *,
    case_id: str = "beamdomino/canonical",
    scene: str = "beamdomino",
    modules: str = "[causal_step]",
) -> None:
    reference = case_dir / "reference"
    reference.mkdir(parents=True)
    (reference / "reference.mp4").write_text("tiny reference\n", encoding="utf-8")
    (reference / "baseline.json").write_text("{}\n", encoding="utf-8")
    (reference / "checksums.sha256").write_text("", encoding="utf-8")
    (case_dir / "case.yaml").write_text(
        "schema_version: 1\n"
        f"case_id: {case_id}\n"
        f"scene: {scene}\n"
        "display_name: Test Case\n"
        f"modules: {modules}\n"
        "reference:\n"
        "  video: reference/reference.mp4\n"
        "  baseline: reference/baseline.json\n"
        "  checksums: reference/checksums.sha256\n"
        "protocol:\n"
        "  candidate_normalization: beamdomino_native\n"
        "  min_duration_sec: 1.0\n"
        "  min_duration_ratio: 0.5\n"
        "  duration_threshold_mode: max\n"
        "  nested: {values: [one, two]}\n"
        "evaluator_lock:\n"
        "  release_id: wmeval-lite-v1\n"
        "  code_revision: test\n"
        "  evaluator_fingerprint: test-fingerprint\n"
        "  module_config: {}\n"
        "  artifact_schema_versions: {}\n"
        "  dependency_versions: {}\n",
        encoding="utf-8",
    )


def _mutate_case_yaml(
    case_dir: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    case_yaml = case_dir / "case.yaml"
    raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    mutate(raw)
    case_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def test_official_cases_and_scene_modules_are_exact() -> None:
    assert ALLOWED_CASES == frozenset(
        {
            "beamdomino/canonical",
            "beamdomino/glancing",
            "beamdomino/direct_strike",
            "liquid_container/rest",
            "liquid_container/step_hold",
            "liquid_container/impulse_decay",
            "liquid_container/chirp_strafe",
        }
    )
    assert SCENE_MODULES == {
        "beamdomino": ("causal_step",),
        "liquid_container": (
            "liquid_velocity",
            "liquid_volume_conservation",
            "liquid_surface_orientation",
            "slosh_spectrum",
        ),
    }


def test_load_case_resolves_reference_paths() -> None:
    case = load_case(FIXTURE_ROOT / "beamdomino" / "canonical")

    assert case.case_id == "beamdomino/canonical"
    assert case.scene == "beamdomino"
    assert case.modules == ("causal_step",)
    assert case.reference_video == (
        FIXTURE_ROOT / "beamdomino" / "canonical" / "reference" / "reference.mp4"
    ).resolve()
    assert case.reference_video.is_absolute()
    assert case.artifacts_dir == case.root / "reference" / "artifacts"


def test_load_case_metadata_is_deeply_read_only() -> None:
    case = load_case(FIXTURE_ROOT / "beamdomino" / "canonical")

    with pytest.raises(FrozenInstanceError):
        case.display_name = "Changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        case.protocol["candidate_normalization"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        case.protocol["nested"]["values"][0] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        case.evaluator_lock["module_config"]["causal_step"]["judge"] = "changed"  # type: ignore[index]


def test_discover_cases_is_sorted_by_case_id() -> None:
    cases = discover_cases(FIXTURE_ROOT)

    assert [case.case_id for case in cases] == sorted(case.case_id for case in cases)


@pytest.mark.parametrize(
    ("case_id", "scene", "modules", "message"),
    [
        ("beamdomino/unknown", "beamdomino", "[causal_step]", "unsupported official Lite case_id"),
        ("beamdomino/canonical", "liquid_container", "[causal_step]", "does not match scene"),
        ("beamdomino/canonical", "beamdomino", "[]", "modules must be"),
    ],
)
def test_load_case_rejects_non_official_metadata(
    tmp_path: Path,
    case_id: str,
    scene: str,
    modules: str,
    message: str,
) -> None:
    case_dir = tmp_path / "case"
    _write_case(case_dir, case_id=case_id, scene=scene, modules=modules)

    with pytest.raises(ValueError, match=message):
        load_case(case_dir)


def test_load_case_reports_missing_required_field(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_case(case_dir)
    case_yaml = case_dir / "case.yaml"
    case_yaml.write_text(
        case_yaml.read_text(encoding="utf-8").replace("display_name: Test Case\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required key 'display_name'"):
        load_case(case_dir)


@pytest.mark.parametrize(
    "key",
    [
        "candidate_normalization",
        "min_duration_sec",
        "min_duration_ratio",
        "duration_threshold_mode",
    ],
)
def test_load_case_rejects_missing_common_protocol_key(
    tmp_path: Path, key: str
) -> None:
    case_dir = tmp_path / "case"
    _write_case(case_dir)
    _mutate_case_yaml(case_dir, lambda raw: raw["protocol"].pop(key))

    with pytest.raises(ValueError, match=rf"protocol missing required key '{key}'"):
        load_case(case_dir)


@pytest.mark.parametrize(
    "key",
    [
        "release_id",
        "code_revision",
        "evaluator_fingerprint",
        "module_config",
        "artifact_schema_versions",
        "dependency_versions",
    ],
)
def test_load_case_rejects_missing_common_evaluator_lock_key(
    tmp_path: Path, key: str
) -> None:
    case_dir = tmp_path / "case"
    _write_case(case_dir)
    _mutate_case_yaml(case_dir, lambda raw: raw["evaluator_lock"].pop(key))

    with pytest.raises(ValueError, match=rf"evaluator_lock missing required key '{key}'"):
        load_case(case_dir)


@pytest.mark.parametrize(
    "key",
    ["module_config", "artifact_schema_versions", "dependency_versions"],
)
def test_load_case_rejects_malformed_evaluator_lock_mapping(
    tmp_path: Path, key: str
) -> None:
    case_dir = tmp_path / "case"
    _write_case(case_dir)
    _mutate_case_yaml(case_dir, lambda raw: raw["evaluator_lock"].update({key: []}))

    with pytest.raises(ValueError, match=rf"evaluator_lock\.{key} must contain a mapping"):
        load_case(case_dir)


def test_validate_case_bundle_accepts_fixture() -> None:
    validate_case_bundle(load_case(FIXTURE_ROOT / "beamdomino" / "canonical"))


def test_validate_case_bundle_rejects_lfs_pointer(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_case(case_dir)
    (case_dir / "reference" / "reference.mp4").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 123\n",
        encoding="utf-8",
    )

    with pytest.raises(CaseValidationError, match="Git LFS object is missing"):
        validate_case_bundle(load_case(case_dir), verify_checksums=False)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("A" * 64 + "  reference/reference.mp4\n", "invalid checksum line"),
        ("0" * 64 + " reference/reference.mp4\n", "invalid checksum line"),
        ("0" * 64 + "   reference/reference.mp4\n", "invalid checksum line"),
        ("0" * 64 + "  ../outside.txt\n", "escapes case root"),
    ],
)
def test_validate_case_bundle_rejects_invalid_checksum_manifest(
    tmp_path: Path, manifest: str, message: str
) -> None:
    case_dir = tmp_path / "case"
    _write_case(case_dir)
    (case_dir / "reference" / "checksums.sha256").write_text(manifest, encoding="utf-8")

    with pytest.raises(CaseValidationError, match=message):
        validate_case_bundle(load_case(case_dir), verify_checksums=False)


def test_validate_case_bundle_rejects_declared_path_outside_case_root(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    _write_case(case_dir)
    outside = tmp_path / "outside.mp4"
    outside.write_text("outside\n", encoding="utf-8")
    case_yaml = case_dir / "case.yaml"
    case_yaml.write_text(
        case_yaml.read_text(encoding="utf-8").replace(
            "video: reference/reference.mp4", "video: ../outside.mp4"
        ),
        encoding="utf-8",
    )

    with pytest.raises(CaseValidationError, match="declared reference path escapes case root"):
        validate_case_bundle(load_case(case_dir), verify_checksums=False)


def test_validate_case_bundle_rejects_checksum_mismatch(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_case(case_dir)
    (case_dir / "reference" / "checksums.sha256").write_text(
        "0" * 64 + "  reference/reference.mp4\n",
        encoding="utf-8",
    )

    with pytest.raises(CaseValidationError, match="checksum mismatch"):
        validate_case_bundle(load_case(case_dir))
