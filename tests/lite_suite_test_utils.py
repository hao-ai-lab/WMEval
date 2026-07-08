"""Shared builders for the WMEval Lite suite-layer tests.

Builds a miniature two-dimension suite (one registry/reference dimension, one
vlm_consistency/reference-free dimension, plus an optional reference-free
registry dimension) with valid v2 case bundles and checksums, and a fake
candidate tree. Video probing/decoding is mocked the same way as the ported
PR #19 tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from evaluation.context import VideoMetadata

RUBRIC_SCENE_NAME = "SceneB_first"
RUBRIC_FILENAME = "openai_gpt-5-4_v1.json"


def write_checksums(case_dir: Path, relative_paths: list[str]) -> None:
    lines = []
    for relative_path in relative_paths:
        digest = hashlib.sha256((case_dir / relative_path).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative_path}")
    (case_dir / "reference" / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _base_case(case_id: str, scene: str, modules: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "case_id": case_id,
        "scene": scene,
        "display_name": case_id,
        "modules": modules,
        "protocol": {
            "candidate_normalization": "none",
            "min_duration_sec": 1.0,
            "min_duration_ratio": 0.5,
            "duration_threshold_mode": "max",
        },
        "evaluator_lock": {
            "release_id": "wmeval-lite-test",
            "code_revision": "test",
            "evaluator_fingerprint": "test-fingerprint",
            "module_config": {},
            "artifact_schema_versions": {},
            "dependency_versions": {},
        },
    }


def write_reference_case(
    cases_root: Path,
    case_id: str = "scene_a/main",
    *,
    modules: list[str] | None = None,
    prompt: str = "A chrome ball rolls between yellow spheres.",
    module_params: dict[str, Any] | None = None,
) -> Path:
    scene = case_id.split("/")[0]
    case_dir = cases_root / case_id
    reference = case_dir / "reference"
    reference.mkdir(parents=True)
    (reference / "reference.mp4").write_text("tiny reference\n", encoding="utf-8")
    raw = _base_case(case_id, scene, modules if modules is not None else ["lpips"])
    raw["prompt"] = prompt
    raw["reference"] = {
        "video": "reference/reference.mp4",
        "checksums": "reference/checksums.sha256",
    }
    if module_params:
        raw["module_params"] = module_params
    write_checksums(case_dir, ["reference/reference.mp4"])
    (case_dir / "case.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    return case_dir


def write_reference_free_case(
    cases_root: Path,
    case_id: str = "scene_b/first",
    *,
    modules: list[str] | None = None,
    prompt: str = "A red car drives around a track.",
    rubric_scene_name: str = RUBRIC_SCENE_NAME,
    module_params: dict[str, Any] | None = None,
) -> Path:
    scene = case_id.split("/")[0]
    case_dir = cases_root / case_id
    reference = case_dir / "reference"
    reference.mkdir(parents=True)
    (reference / "ref.png").write_text("tiny first frame\n", encoding="utf-8")
    raw = _base_case(case_id, scene, modules if modules is not None else [])
    raw["prompt"] = prompt
    raw["perspective"] = "first_person"
    raw["rubric_scene_name"] = rubric_scene_name
    raw["reference"] = {
        "first_frame": "reference/ref.png",
        "checksums": "reference/checksums.sha256",
    }
    if module_params:
        raw["module_params"] = module_params
    write_checksums(case_dir, ["reference/ref.png"])
    (case_dir / "case.yaml").write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    return case_dir


DEFAULT_SUITE_YAML: dict[str, Any] = {
    "suite_name": "WMEval Lite Test",
    "version": "v-test",
    "description": "toy suite",
    "dimensions": {
        "toy_metric": {
            "display_name": "Toy Metric",
            "evaluator": "registry",
            "requires_reference": True,
            "modules": ["lpips"],
            "metric_config": {"eval_fps": 4.0, "max_pairs": 8},
            "cases": ["scene_a/main"],
            "aggregate": {
                "headline": "lpips_mean",
                "direction": "lower",
                "extra_means": ["lpips_median"],
            },
        },
        "toy_promptcons": {
            "display_name": "Toy Prompt Consistency",
            "evaluator": "vlm_consistency",
            "requires_reference": False,
            "vlm": {"model": "gpt-5.4"},
            "cases": ["scene_b/first"],
            "aggregate": {"headline": "overall_score", "direction": "higher"},
        },
    },
    "legacy_case_aliases": {"toy_metric": {"old_scene_a": "scene_a/main"}},
}


def write_suite(
    suite_root: Path, suite_yaml: dict[str, Any] | None = None
) -> Path:
    payload = suite_yaml if suite_yaml is not None else DEFAULT_SUITE_YAML
    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / "suite.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return suite_root


def build_default_suite_dir(suite_root: Path) -> Path:
    """suite.yaml + both default cases, ready for load_suite/run_suite."""
    write_suite(suite_root)
    write_reference_case(suite_root / "cases")
    write_reference_free_case(suite_root / "cases")
    return suite_root


def write_candidates(model_root: Path, case_ids: list[str], seed: str = "0001") -> None:
    for case_id in case_ids:
        video = model_root / case_id / f"{seed}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_text(f"fake candidate {case_id} {seed}\n", encoding="utf-8")


def regenerate_checksums(case_dir: Path) -> None:
    """Rewrite the case's checksum manifest over everything under reference/."""
    reference_root = case_dir / "reference"
    checksums_path = reference_root / "checksums.sha256"
    lines = []
    for path in sorted(reference_root.rglob("*")):
        if not path.is_file() or path == checksums_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(case_dir).as_posix()}")
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rubric(
    case_dir: Path, scene_name: str = RUBRIC_SCENE_NAME
) -> Path:
    """Write a rubric into the case bundle (case-relative rubric cache) and
    refresh the checksum manifest to keep the bundle valid."""
    rubric = {
        "scene_name": scene_name,
        "prompt_text": "A red car drives around a track.",
        "rubric_items": [
            {
                "rubric_id": "env_01",
                "agent": "environment",
                "weight": 1.0,
                "rubric_text": "Is there a track?",
            },
            {
                "rubric_id": "obj_01",
                "agent": "object",
                "weight": 1.0,
                "rubric_text": "Is the car red?",
            },
        ],
        "metadata": {
            "generator_provider": "openai",
            "generator_model": "gpt-5.4",
            "generated_at": "2026-01-01T00:00:00Z",
            "num_candidates_generated": 2,
        },
    }
    path = (
        case_dir
        / "reference"
        / "artifacts"
        / "rubrics"
        / scene_name
        / RUBRIC_FILENAME
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rubric, indent=2), encoding="utf-8")
    regenerate_checksums(case_dir)
    return path


def mock_video_probing(monkeypatch: pytest.MonkeyPatch, duration_sec: float = 10.0) -> None:
    from evaluation.lite import validation

    def metadata(path: str) -> VideoMetadata:
        return VideoMetadata(
            path=str(path),
            fps=10.0,
            frame_count=int(duration_sec * 10),
            width=832,
            height=464,
            duration_sec=duration_sec,
        )

    monkeypatch.setattr(validation.video_io, "get_video_metadata", metadata)
    monkeypatch.setattr(
        validation.video_io,
        "load_frames_by_indices",
        lambda path, indices: [object() for _ in indices],
    )


class FakeJudgeResult:
    """Duck-typed JudgeResult substitute (model_dump + attribute access)."""

    def __init__(self, overall: float, env: float, obj: float) -> None:
        self.overall_score = overall
        self.environment_result = type("R", (), {"score": env})()
        self.object_result = type("R", (), {"score": obj})()
        self._payload = {
            "overall_score": overall,
            "environment_result": {"score": env},
            "object_result": {"score": obj},
        }

    def model_dump(self) -> dict[str, Any]:
        return dict(self._payload)
