"""Scoring the synthetic trajectory set (``--split synthetic|full``).

Hermetic tests build a toy suite with one conditioning core case and two
synthetic trajectories; the data-dependent ones check the real suite.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from evaluation.lite.aggregate import build_aggregate
from evaluation.lite.output import PER_CASE_RESULTS_FILENAME, read_rows
from evaluation.lite.suite import run_suite
from evaluation.lite.suite_loader import SuiteValidationError, load_suite
from evaluation.lite.synthetic import (
    SyntheticSuiteError,
    load_synthetic_cases,
    validate_synthetic_case,
)
from evaluation.lite.validation import CaseValidationError
from lite_suite_test_utils import (
    DEFAULT_SUITE_YAML,
    mock_video_probing,
    write_candidates,
    write_reference_case,
    write_suite,
)

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "toy_family"
TRAJECTORIES = ("s00_push", "s01_roll")


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        # self-match on the reference itself scores 0, anything else 0.4
        score = 0.0 if kwargs["video_path"] == kwargs["ref_video_path"] else 0.4
        return {"results": {"lpips": {"lpips_mean": score, "lpips_median": score}}}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_suite(tmp_path: Path, *, overrides: dict | None = None) -> Path:
    payload = copy.deepcopy(DEFAULT_SUITE_YAML)
    del payload["dimensions"]["toy_promptcons"]
    payload["synthetic"] = {"module_params": overrides or {}}
    root = write_suite(tmp_path / "suite", payload)
    write_reference_case(
        root / "cases",
        module_params={"lpips": {"lpips_reference_events_json": "reference/artifacts/contact_events.json"}},
    )
    rows, references = [], {}
    for index, slug in enumerate(TRAJECTORIES):
        rel = f"synthetic/{FAMILY}/scene_a__main/{slug}"
        traj = root / rel
        (traj / "reference/artifacts").mkdir(parents=True)
        (traj / "meta.json").write_text("{}\n", encoding="utf-8")
        (traj / "reference/reference.mp4").write_text(f"reference {slug}\n", encoding="utf-8")
        (traj / "reference/artifacts/contact_events.json").write_text("{}\n", encoding="utf-8")
        rows.append({
            "synthetic_id": rel, "path": rel, "family": FAMILY, "scene": "scene_a",
            "pattern": slug, "conditioning_case": "scene_a/main", "action_target": "ball",
            "perspective": "third_person", "prompt": f"Scene a. The ball moves ({slug}).",
            "first_frame": "cases/scene_a/main/reference/reference.mp4",
            "dimensions": ["toy_metric", "not_in_this_suite"],
            "reference_status": "unity_rendered",
            "reference_quality": "weak_first_frame" if index else "ok",
            "files": {"meta.json": _sha(traj / "meta.json")},
        })
        references[rel] = {"files": {
            name: _sha(traj / name)
            for name in ("reference/reference.mp4", "reference/artifacts/contact_events.json")
        }}
    (root / "synthetic/manifest.json").write_text(json.dumps({"trajectories": rows}), encoding="utf-8")
    (root / "synthetic/references.json").write_text(json.dumps({"references": references}), encoding="utf-8")
    return root


def _manifest(path: Path, root: Path, *, cross: bool = False) -> Path:
    lines = []
    for index, slug in enumerate(TRAJECTORIES):
        source = TRAJECTORIES[1 - index] if cross else slug
        lines.append(json.dumps({
            "model": "selfmatch", "case_id": f"synthetic/{FAMILY}/scene_a__main/{slug}", "seed": "0",
            "candidate_video": str(root / f"synthetic/{FAMILY}/scene_a__main/{source}/reference/reference.mp4"),
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_synthetic_case_inherits_conditioning_case(tmp_path: Path) -> None:
    root = _toy_suite(tmp_path)
    entries = load_synthetic_cases(load_suite(root))
    assert sorted(entries) == [f"synthetic/{FAMILY}/scene_a__main/{s}" for s in TRAJECTORIES]
    entry = entries[f"synthetic/{FAMILY}/scene_a__main/s00_push"]
    case = entry.case
    assert entry.dimensions == ("toy_metric",)  # undeclared-by-suite dimension dropped
    assert case.root == (root / entry.trajectory.path).resolve()
    assert case.reference_video == case.root / "reference/reference.mp4"
    assert case.prompt == "Scene a. The ball moves (s00_push)."
    assert case.protocol["candidate_normalization"] == "none"
    assert case.evaluator_lock["evaluator_fingerprint"] == "test-fingerprint"
    assert case.action_target == "ball"
    assert case.first_frame is None  # a video, not an image
    # artifact paths re-resolve against the trajectory, not the core bundle
    assert case.module_params["lpips"]["lpips_reference_events_json"] == str(
        case.root / "reference/artifacts/contact_events.json"
    )
    assert entry.row_fields() == {
        "split": "synthetic", "family": FAMILY,
        "conditioning_case": "scene_a/main", "reference_quality": "ok",
    }
    validate_synthetic_case(entry)


def test_module_param_overrides_and_validation(tmp_path: Path) -> None:
    root = _toy_suite(tmp_path, overrides={"scene_a/main": {"lpips": {
        "lpips_reference_events_json": "reference/artifacts/other.json", "lpips_vlm_model": "gpt-x"}}})
    entry = next(iter(load_synthetic_cases(load_suite(root)).values()))
    assert entry.case.module_params["lpips"]["lpips_vlm_model"] == "gpt-x"
    with pytest.raises(CaseValidationError, match="missing reference input"):
        validate_synthetic_case(entry)


def test_overrides_are_whitelisted(tmp_path: Path) -> None:
    with pytest.raises(SuiteValidationError, match="whitelist"):
        load_suite(_toy_suite(tmp_path, overrides={"scene_a/main": {"lpips": {"bogus_key": 1}}}))


def test_checksums_and_unlisted_reference_files(tmp_path: Path) -> None:
    root = _toy_suite(tmp_path)
    entry = next(iter(load_synthetic_cases(load_suite(root)).values()))
    extra = entry.case.root / "reference/artifacts/unlisted.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(CaseValidationError, match="not in references.json"):
        validate_synthetic_case(entry)
    extra.unlink()
    entry.case.reference_video.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CaseValidationError, match="checksum mismatch"):
        validate_synthetic_case(entry)
    validate_synthetic_case(entry, verify_checksums=False)


def test_not_applicable_dimension_is_skipped(tmp_path: Path) -> None:
    root = _toy_suite(tmp_path)
    marker = root / f"synthetic/{FAMILY}/scene_a__main/s01_roll/reference/artifacts/dimension_applicability.json"
    marker.write_text(json.dumps({"not_applicable": {"toy_metric": "no chain"}}), encoding="utf-8")
    entry = load_synthetic_cases(load_suite(root))[f"synthetic/{FAMILY}/scene_a__main/s01_roll"]
    assert entry.dimensions == ()
    assert dict(entry.not_applicable) == {"toy_metric": "no chain"}


def test_suite_without_synthetic_section(tmp_path: Path) -> None:
    payload = copy.deepcopy(DEFAULT_SUITE_YAML)
    del payload["dimensions"]["toy_promptcons"]
    root = write_suite(tmp_path / "suite", payload)
    write_reference_case(root / "cases")
    with pytest.raises(SyntheticSuiteError, match="no synthetic split"):
        load_synthetic_cases(load_suite(root))


def test_run_synthetic_split_self_match_and_separate_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _toy_suite(tmp_path)
    mock_video_probing(monkeypatch)
    runner = RecordingRunner()
    aggregate = run_suite(
        suite_root=root, output_dir=tmp_path / "out", split="full",
        candidate_manifest=_manifest(tmp_path / "self.jsonl", root),
        runner=runner, log=lambda _m: None,
    )
    rows = read_rows(tmp_path / "out" / PER_CASE_RESULTS_FILENAME)
    assert [r["status"] for r in rows] == ["completed", "completed"]
    assert {r["split"] for r in rows} == {"synthetic"}
    assert {r["reference_quality"] for r in rows} == {"ok", "weak_first_frame"}
    assert all(c["lpips_reference_events_json"].endswith("contact_events.json") for c in runner.calls)
    # core headline untouched (no core candidates), synthetic reported apart
    assert aggregate["models"] == {} and aggregate["num_rows"] == 0
    synthetic = aggregate["synthetic"]["models"]["selfmatch"]["dimensions"]["toy_metric"]
    assert synthetic["headline"]["value"] == 0.0
    assert synthetic["headline"]["n_trajectories"] == 2
    assert synthetic["headline_reference_ok"]["n_trajectories"] == 1
    assert synthetic["by_family"][FAMILY]["value"] == 0.0
    assert (tmp_path / "out" / "aggregate_summary_synthetic.csv").is_file()


def test_cross_match_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _toy_suite(tmp_path)
    mock_video_probing(monkeypatch)
    aggregate = run_suite(
        suite_root=root, output_dir=tmp_path / "out", split="synthetic",
        candidate_manifest=_manifest(tmp_path / "cross.jsonl", root, cross=True),
        runner=RecordingRunner(), log=lambda _m: None,
    )
    headline = aggregate["synthetic"]["models"]["selfmatch"]["dimensions"]["toy_metric"]["headline"]
    assert headline["value"] == pytest.approx(0.4)


def test_core_split_ignores_synthetic_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _toy_suite(tmp_path)
    mock_video_probing(monkeypatch)
    model_root = tmp_path / "model"
    write_candidates(model_root, ["scene_a/main", f"synthetic/{FAMILY}/scene_a__main/s00_push"])
    runner = RecordingRunner()
    aggregate = run_suite(suite_root=root, output_dir=tmp_path / "core", model_root=model_root,
                          runner=runner, log=lambda _m: None)
    assert len(runner.calls) == 1 and "synthetic" not in aggregate
    assert aggregate["models"]["model"]["dimensions"]["toy_metric"]["num_rows"] == 1
    # full: same core numbers plus the synthetic block from directory discovery
    full = run_suite(suite_root=root, output_dir=tmp_path / "full", model_root=model_root,
                     runner=RecordingRunner(), split="full", log=lambda _m: None)
    assert full["models"] == aggregate["models"]
    assert full["synthetic"]["models"]["model"]["dimensions"]["toy_metric"]["num_trajectories"] == 1


def test_aggregate_keeps_synthetic_rows_out_of_core(tmp_path: Path) -> None:
    suite = load_suite(_toy_suite(tmp_path))
    core = {"dimension": "toy_metric", "case_id": "scene_a/main", "model": "m", "seed": "0",
            "status": "completed", "metrics": {"lpips_mean": 0.2}}
    synthetic = {**core, "case_id": f"synthetic/{FAMILY}/scene_a__main/s00_push", "split": "synthetic",
                 "family": FAMILY, "reference_quality": "ok", "metrics": {"lpips_mean": 0.9}}
    aggregate = build_aggregate(suite, [core, synthetic])
    assert aggregate["models"]["m"]["dimensions"]["toy_metric"]["headline"]["value"] == 0.2
    assert aggregate["num_rows"] == 1
    assert aggregate["synthetic"]["models"]["m"]["dimensions"]["toy_metric"]["headline"]["value"] == 0.9


def test_synthetic_capture_segments() -> None:
    from tools.dataset.detect_beamdomino_segments import detect_segments, log_from_synthetic_capture

    def frame(t: float, ball_z: float, roof_tilt: float, fallen: bool) -> dict:
        import math

        half = math.radians(roof_tilt) / 2
        domino_half = math.radians(80 if fallen else 0) / 2
        return {
            "time_sec": t, "target_pos": [0.0, 0.55, ball_z], "target_velocity": [0.0, 0.0, 4.0],
            "target_rot": [0, 0, 0, 1],
            "objects": {
                "Pillar": {"pos": [0.0, 0.6, 0.0], "rot": [0, 0, 0, 1],
                           "bounds_center": [0.0, 0.6, 0.0], "bounds_extents": [0.175, 0.6, 0.175]},
                "Roof": {"pos": [0.0, 1.2, 0.5], "rot": [math.sin(half), 0, 0, math.cos(half)],
                         "bounds_center": [0.0, 1.2, 0.5], "bounds_extents": [1.0, 0.075, 1.0]},
                "Domino_S00_D00": {"pos": [0.0, 0.55, 5.0], "rot": [math.sin(domino_half), 0, 0, math.cos(domino_half)],
                                   "bounds_center": [0.0, 0.55, 5.0], "bounds_extents": [0.35, 0.55, 0.09]},
            },
        }

    frames = [frame(i / 10, -3.0 + 0.4 * i, 20.0 if i >= 9 else 0.0, i >= 14) for i in range(7)]
    frames += [frame(i / 10, -0.6, 20.0 if i >= 9 else 0.0, i >= 14) for i in range(7, 30)]
    capture = {"synthetic_id": "toy", "frames": frames}
    segments = detect_segments(log_from_synthetic_capture(capture))
    assert segments["trajectory_shape"] == "canonical"
    assert [e["name"] for e in segments["events"]] == ["rolling", "impact", "topple", "cascade"]


@pytest.mark.integration("synthetic_data")
def test_real_suite_synthetic_cases_build() -> None:
    suite = load_suite(ROOT / "lite")
    entries = load_synthetic_cases(suite)
    assert len(entries) == 220
    for entry in entries.values():
        case = entry.case
        assert case.reference_video.is_file()
        assert set(entry.dimensions) | set(entry.not_applicable) <= set(entry.trajectory.dimensions)
        if "qa_dataset" in case.reference_artifacts:
            assert case.module_params["causal_step"]["causal_step_video_id"]
        if "prompt_scene_consistency" in entry.dimensions:
            assert case.first_frame is not None and case.first_frame.is_file()
    by_family = {}
    for entry in entries.values():
        by_family.setdefault(entry.trajectory.family, set()).update(entry.dimensions)
    assert by_family["camera_orbit"] == {"lighting_shading"}
    assert by_family["liquid_container"] == {"fluid_mechanics", "lpips"}


# Reference inputs authored with the hosted judge (tools/dataset/build_synthetic_reference_inputs.py).
JUDGE_AUTHORED = ("contact_events.json", "qa_dataset.json", "rubric for")


@pytest.mark.integration("synthetic_data")
def test_real_suite_synthetic_bundles_validate() -> None:
    suite = load_suite(ROOT / "lite")
    pending = []
    for case_id, entry in load_synthetic_cases(suite).items():
        try:
            validate_synthetic_case(entry, verify_checksums=False)
        except CaseValidationError as exc:
            if not any(name in str(exc) for name in JUDGE_AUTHORED):
                raise
            pending.append(f"{case_id}: {exc}")
    if pending:
        pytest.skip(f"{len(pending)} trajectory(ies) still lack judge-authored reference inputs, e.g. {pending[0]}")
