from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from evaluation.lite.output import PER_CASE_RESULTS_FILENAME, read_rows, write_rows
from evaluation.lite.suite import SuiteRunError, run_suite
from tests.lite_suite_test_utils import (
    DEFAULT_SUITE_YAML,
    FakeJudgeResult,
    build_default_suite_dir,
    mock_video_probing,
    write_candidates,
    write_reference_case,
    write_reference_free_case,
    write_rubric,
    write_suite,
)


class RecordingRunner:
    def __init__(self, report: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.report = report or {
            "results": {"lpips": {"lpips_mean": 0.3, "lpips_median": 0.25}}
        }

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.report


def _quiet(_message: str) -> None:
    pass


def _registry_only_suite(tmp_path: Path) -> Path:
    payload = copy.deepcopy(DEFAULT_SUITE_YAML)
    del payload["dimensions"]["toy_promptcons"]
    root = write_suite(tmp_path / "suite", payload)
    write_reference_case(root / "cases")
    return root


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
def test_orbit_receipt_is_consumed_and_resume_invalidates_video_or_params(tmp_path, monkeypatch):
    suite_root = _registry_only_suite(tmp_path)
    case_path = suite_root/'cases/scene_a/main/case.yaml'
    raw = yaml.safe_load(case_path.read_text())
    raw['protocol']['candidate_normalization'] = 'lighting_orbit_10s_fps12'
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    mock_video_probing(monkeypatch)
    video = tmp_path/'rollout.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=12:duration=10',
                    '-c:v','libx264',str(video)],check=True,capture_output=True)
    mapping = {'schema_version':1,'native_fps':12,'canonical_zero_frame':0,
               'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest()}
    entry = {'model':'test','case_id':'scene_a/main','seed':'42','candidate_video':str(video),
             'params':{'lighting_time_mapping':mapping}}
    manifest = tmp_path/'manifest.jsonl'
    runner = RecordingRunner()
    def evaluate():
        manifest.write_text(json.dumps(entry)+'\n')
        run_suite(suite_root=suite_root,output_dir=tmp_path/'out',candidate_manifest=manifest,
                  runner=runner,resume=True,log=_quiet)
        return read_rows(tmp_path/'out'/PER_CASE_RESULTS_FILENAME)
    assert evaluate()[0]['status']=='completed'
    evaluate()
    assert len(runner.calls)==1
    assert 'lighting_time_mapping' not in runner.calls[0]
    from evaluation.lite.normalization import _probe
    assert _probe(Path(runner.calls[0]['video_path']),timeout_sec=30)['frames']==120
    entry['params']['max_pairs']=2
    assert evaluate()[0]['status']=='completed' and len(runner.calls)==2
    # A same-path replacement must invalidate the saved result. Until the
    # receipt is updated, fail rather than skipping or silently rescoring.
    with video.open('ab') as handle: handle.write(b'\0\0\0\0')
    assert evaluate()[0]['status']=='error' and len(runner.calls)==2
    mapping['video_sha256']=hashlib.sha256(video.read_bytes()).hexdigest()
    rows=evaluate()
    assert len(rows)==1 and rows[0]['status']=='completed' and len(runner.calls)==3
    assert rows[0]['input_contract']['lighting_time_mapping']=='receipt'


def test_run_suite_registry_dimension_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_root = _registry_only_suite(tmp_path)
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_a/main"])
    mock_video_probing(monkeypatch)
    runner = RecordingRunner()

    aggregate = run_suite(
        suite_root=suite_root,
        output_dir=tmp_path / "out",
        model_root=model_root,
        runner=runner,
        run_id="test-run",
        log=_quiet,
    )

    assert len(runner.calls) == 1
    kwargs = runner.calls[0]
    assert kwargs["modules"] == ("lpips",)
    assert kwargs["video_path"].endswith("scene_a/main/0001.mp4")
    assert kwargs["ref_video_path"].endswith("reference/reference.mp4")
    assert kwargs["text_prompt"] == "A chrome ball rolls between yellow spheres."
    assert kwargs["eval_fps"] == 4.0
    assert kwargs["max_pairs"] == 8

    rows = read_rows(tmp_path / "out" / PER_CASE_RESULTS_FILENAME)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"
    assert row["dimension"] == "toy_metric"
    assert row["case_id"] == "scene_a/main"
    assert row["model"] == "genie3"
    assert row["seed"] == "0001"
    assert row["metrics"]["lpips_mean"] == 0.3
    assert row["evaluator_lock"]["release_id"] == "wmeval-lite-test"
    assert row["reference_video"].endswith("reference/reference.mp4")

    summary = aggregate["models"]["genie3"]["dimensions"]["toy_metric"]
    assert summary["headline"] == {
        "metric": "lpips_mean",
        "value": 0.3,
        "std_across_cases": None,  # single case — no fabricated spread
        "direction": "lower",
    }
    assert (tmp_path / "out" / "aggregate_summary.json").is_file()
    assert (tmp_path / "out" / "aggregate_summary.csv").is_file()


def test_run_suite_reference_free_registry_uses_candidate_as_dummy_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = copy.deepcopy(DEFAULT_SUITE_YAML)
    del payload["dimensions"]["toy_promptcons"]
    payload["dimensions"]["toy_shadow"] = {
        "display_name": "Toy Shadow",
        "evaluator": "registry",
        "requires_reference": False,
        "modules": ["shadow_consistency"],
        "metric_config": {},
        "cases": ["scene_b/first"],
        "aggregate": {
            "headline": "shadow_direction_jitter_deg_per_frame",
            "direction": "lower",
        },
    }
    del payload["dimensions"]["toy_metric"]
    del payload["legacy_case_aliases"]
    root = write_suite(tmp_path / "suite", payload)
    write_reference_free_case(
        root / "cases",
        modules=["shadow_consistency"],
        module_params={
            "shadow_consistency": {
                "shadow_text_prompt": "car.",
                "shadow_anchor_label": "car",
            }
        },
    )
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_b/first"])
    mock_video_probing(monkeypatch)
    runner = RecordingRunner(
        report={
            "results": {
                "shadow_consistency": {
                    "scene_retention_rate_pct": 98.0,
                    "shadow_detection_rate_pct": 88.0,
                    "multi_object_sun_disagreement_deg": 4.0,
                    "shadow_direction_jitter_deg_per_frame": 1.5,
                    "shadow_area_stability_pct": 6.0,
                    "shadow_shape_drift": 0.02,
                }
            }
        }
    )

    run_suite(
        suite_root=root,
        output_dir=tmp_path / "out",
        model_root=model_root,
        runner=runner,
        log=_quiet,
    )

    kwargs = runner.calls[0]
    # dummy ref: candidate stands in for the missing reference
    assert kwargs["ref_video_path"] == kwargs["video_path"]
    # case-level module_params flow through the whitelist merge
    assert kwargs["shadow_text_prompt"] == "car."
    assert kwargs["shadow_anchor_label"] == "car"

    row = read_rows(tmp_path / "out" / PER_CASE_RESULTS_FILENAME)[0]
    assert row["reference_video"] is None
    assert row["metrics"]["shadow_direction_jitter_deg_per_frame"] == 1.5


def test_run_suite_candidate_params_override_and_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_root = _registry_only_suite(tmp_path)
    mock_video_probing(monkeypatch)
    runner = RecordingRunner()
    video = tmp_path / "rollout.mp4"
    video.write_text("fake\n", encoding="utf-8")

    manifest = tmp_path / "candidates.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "model": "genie3",
                "case_id": "scene_a/main",
                "seed": "0001",
                "candidate_video": str(video),
                "params": {"max_pairs": 4},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_suite(
        suite_root=suite_root,
        output_dir=tmp_path / "out",
        candidate_manifest=manifest,
        runner=runner,
        log=_quiet,
    )
    assert runner.calls[0]["max_pairs"] == 4  # candidate params win

    # An unwhitelisted candidate param becomes an error row, not a crash.
    manifest.write_text(
        json.dumps(
            {
                "model": "genie3",
                "case_id": "scene_a/main",
                "seed": "0002",
                "candidate_video": str(video),
                "params": {"totally_unknown_kwarg": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_suite(
        suite_root=suite_root,
        output_dir=tmp_path / "out2",
        candidate_manifest=manifest,
        runner=runner,
        log=_quiet,
    )
    row = read_rows(tmp_path / "out2" / PER_CASE_RESULTS_FILENAME)[0]
    assert row["status"] == "error"
    assert "totally_unknown_kwarg" in row["error"]


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
def test_orbit_without_receipt_runs_from_a_model_root(tmp_path, monkeypatch):
    suite_root = _registry_only_suite(tmp_path)
    case_path = suite_root/'cases/scene_a/main/case.yaml'
    raw = yaml.safe_load(case_path.read_text())
    raw['protocol']['candidate_normalization'] = 'lighting_orbit_10s_fps12'
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    mock_video_probing(monkeypatch)
    video = tmp_path/'new_model'/'scene_a'/'main'/'42.mp4'
    video.parent.mkdir(parents=True)
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=16:duration=10.0625',
                    '-c:v','libx264',str(video)],check=True,capture_output=True)
    runner = RecordingRunner()
    run_suite(suite_root=suite_root,output_dir=tmp_path/'out',model_root=tmp_path/'new_model',
              runner=runner,log=_quiet)
    row = read_rows(tmp_path/'out'/PER_CASE_RESULTS_FILENAME)[0]
    assert row['status'] == 'completed'
    assert row['input_contract']['lighting_time_mapping'] == 'derived'
    from evaluation.lite.normalization import _probe
    assert _probe(Path(runner.calls[0]['video_path']),timeout_sec=30)['frames'] == 120


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg required')
def test_derived_orbit_mapping_is_reported_warned_and_counted(tmp_path, monkeypatch):
    suite_root = _registry_only_suite(tmp_path)
    case_path = suite_root/'cases/scene_a/main/case.yaml'
    raw = yaml.safe_load(case_path.read_text())
    raw['protocol']['candidate_normalization'] = 'lighting_orbit_10s_fps12'
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    mock_video_probing(monkeypatch)
    for model, zero, body in (('plain', None, 'model.yaml absent'), ('warm', 4, None)):
        video = tmp_path/model/'scene_a'/'main'/'42.mp4'
        video.parent.mkdir(parents=True)
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=16',
                        '-frames:v',str(160 + (zero or 0)),'-c:v','libx264',str(video)],
                       check=True,capture_output=True)
        if zero is not None:
            (tmp_path/model/'model.yaml').write_text(f'lighting_canonical_zero_frame: {zero}\n')
    messages, details, runners = [], {}, {}
    for model in ('plain', 'warm'):
        runners[model] = RecordingRunner()
        run_suite(suite_root=suite_root,output_dir=tmp_path/'out'/model,model_root=tmp_path/model,
                  runner=runners[model],log=messages.append)
        row = read_rows(tmp_path/'out'/model/PER_CASE_RESULTS_FILENAME)[0]
        assert row['status'] == 'completed' and row['input_contract']['lighting_time_mapping'] == 'derived'
        details[model] = row['input_contract']['lighting_time_mapping_detail']
        # Normalization-only params never reach the runner.
        assert 'lighting_canonical_zero_frame' not in runners[model].calls[0]
        summary = json.loads((tmp_path/'out'/model/'aggregate_summary.json').read_text())
        assert summary['models'][model]['dimensions']['toy_metric']['lighting_time_mapping_counts'] == {'derived': 1}
    assert details['plain'] == {'native_fps': 16, 'native_fps_source': 'container', 'canonical_zero_frame': 0,
                                'canonical_zero_frame_source': 'default', 'container_fps': 16.0, 'retimed': False}
    assert details['warm']['canonical_zero_frame'] == 4
    assert details['warm']['canonical_zero_frame_source'] == 'lighting_canonical_zero_frame'
    warnings = [m for m in messages if 'lighting_time_mapping derived' in m]
    assert len(warnings) == 2
    assert 'plain/42' in warnings[0] and 'if the orbit does not start at the first frame' in warnings[0]
    assert 'warm/42' in warnings[1] and 'if the orbit does not start' not in warnings[1]


def test_lighting_model_params_do_not_break_other_dimensions(tmp_path, monkeypatch):
    suite_root = _registry_only_suite(tmp_path)
    model_root = tmp_path / 'waypoint_like'
    write_candidates(model_root, ['scene_a/main'])
    (model_root / 'model.yaml').write_text('lighting_canonical_zero_frame: 4\n', encoding='utf-8')
    mock_video_probing(monkeypatch)
    runner = RecordingRunner()
    run_suite(suite_root=suite_root, output_dir=tmp_path/'out', model_root=model_root,
              runner=runner, log=_quiet)
    row = read_rows(tmp_path/'out'/PER_CASE_RESULTS_FILENAME)[0]
    assert row['status'] == 'completed'
    assert 'lighting_canonical_zero_frame' not in runner.calls[0]
    assert 'lighting_time_mapping' not in row['input_contract']


def test_run_suite_module_error_reports_error_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_root = _registry_only_suite(tmp_path)
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_a/main"])
    mock_video_probing(monkeypatch)
    runner = RecordingRunner(
        report={"results": {"lpips": {"module": "lpips", "error": "no torch"}}}
    )

    run_suite(
        suite_root=suite_root,
        output_dir=tmp_path / "out",
        model_root=model_root,
        runner=runner,
        log=_quiet,
    )

    row = read_rows(tmp_path / "out" / PER_CASE_RESULTS_FILENAME)[0]
    assert row["status"] == "error"
    assert "lpips: no torch" in row["error"]


def test_run_suite_missing_and_invalid_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_root = _registry_only_suite(tmp_path)
    runner = RecordingRunner()

    missing_manifest = tmp_path / "missing.jsonl"
    missing_manifest.write_text(
        json.dumps(
            {
                "model": "genie3",
                "case_id": "scene_a/main",
                "seed": "0001",
                "candidate_video": str(tmp_path / "nope.mp4"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_suite(
        suite_root=suite_root,
        output_dir=tmp_path / "out_missing",
        candidate_manifest=missing_manifest,
        runner=runner,
        log=_quiet,
    )
    row = read_rows(tmp_path / "out_missing" / PER_CASE_RESULTS_FILENAME)[0]
    assert row["status"] == "missing_candidate"
    assert not runner.calls

    # too-short candidate → invalid_candidate
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_a/main"])
    mock_video_probing(monkeypatch, duration_sec=0.2)
    run_suite(
        suite_root=suite_root,
        output_dir=tmp_path / "out_invalid",
        model_root=model_root,
        runner=runner,
        log=_quiet,
    )
    row = read_rows(tmp_path / "out_invalid" / PER_CASE_RESULTS_FILENAME)[0]
    assert row["status"] == "invalid_candidate"
    assert "severely truncated" in row["error"]
    assert not runner.calls


def test_run_suite_vlm_dimension_via_injected_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluation.lite.adapters.registry_adapter import RegistryAdapter
    from evaluation.lite.adapters.vlm_consistency_adapter import VlmConsistencyAdapter

    suite_root = build_default_suite_dir(tmp_path / "suite")
    write_rubric(suite_root / "cases" / "scene_b" / "first")
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_a/main", "scene_b/first"])
    mock_video_probing(monkeypatch)

    judge_calls: list[dict[str, Any]] = []

    def fake_judge(**kwargs: Any) -> FakeJudgeResult:
        judge_calls.append(kwargs)
        return FakeJudgeResult(0.9, 0.95, 0.85)

    def fake_frames(**kwargs: Any):
        return [("frame_t=0.0s", "b64data")], 1

    runner = RecordingRunner()
    adapters = {
        "registry": RegistryAdapter(runner=runner),
        "vlm_consistency": VlmConsistencyAdapter(
            judge=fake_judge, frame_preparer=fake_frames
        ),
    }

    aggregate = run_suite(
        suite_root=suite_root,
        output_dir=tmp_path / "out",
        model_root=model_root,
        adapters=adapters,
        log=_quiet,
    )

    assert len(judge_calls) == 1
    call = judge_calls[0]
    assert call["provider"] == "openai"
    assert call["model"] == "gpt-5.4"
    assert call["alpha"] == 0.6
    assert call["rubric"].scene_name == "SceneB_first"

    rows = read_rows(tmp_path / "out" / PER_CASE_RESULTS_FILENAME)
    by_dimension = {row["dimension"]: row for row in rows}
    assert by_dimension["toy_promptcons"]["status"] == "completed"
    assert by_dimension["toy_promptcons"]["metrics"] == {
        "overall_score": 0.9,
        "env_score": 0.95,
        "obj_score": 0.85,
    }
    assert by_dimension["toy_metric"]["status"] == "completed"

    promptcons = aggregate["models"]["genie3"]["dimensions"]["toy_promptcons"]
    assert promptcons["headline"]["value"] == 0.9


def test_run_suite_resume_skips_done_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_root = _registry_only_suite(tmp_path)
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_a/main"])
    mock_video_probing(monkeypatch)
    runner = RecordingRunner()

    for _ in range(2):
        run_suite(
            suite_root=suite_root,
            output_dir=tmp_path / "out",
            model_root=model_root,
            runner=runner,
            resume=True,
            log=_quiet,
        )

    assert len(runner.calls) == 1  # second pass resumed
    rows = read_rows(tmp_path / "out" / PER_CASE_RESULTS_FILENAME)
    assert len(rows) == 1


def test_run_suite_resume_retries_error_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_root = _registry_only_suite(tmp_path)
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_a/main"])
    mock_video_probing(monkeypatch)
    runner = RecordingRunner()
    output_dir = tmp_path / "out"

    run_suite(
        suite_root=suite_root,
        output_dir=output_dir,
        model_root=model_root,
        runner=runner,
        resume=True,
        log=_quiet,
    )
    results_path = output_dir / PER_CASE_RESULTS_FILENAME
    rows = read_rows(results_path)
    rows[0]["status"] = "error"
    rows[0]["error"] = "transient GPU failure"
    write_rows(results_path, rows)

    run_suite(
        suite_root=suite_root,
        output_dir=output_dir,
        model_root=model_root,
        runner=runner,
        resume=True,
        log=_quiet,
    )

    assert len(runner.calls) == 2
    rows = read_rows(results_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"


def test_run_suite_resume_replaces_stale_fingerprint_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_root = _registry_only_suite(tmp_path)
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_a/main"])
    mock_video_probing(monkeypatch)
    runner = RecordingRunner()
    output_dir = tmp_path / "out"

    run_suite(
        suite_root=suite_root,
        output_dir=output_dir,
        model_root=model_root,
        runner=runner,
        resume=True,
        log=_quiet,
    )

    case_yaml = suite_root / "cases" / "scene_a" / "main" / "case.yaml"
    raw = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    raw["evaluator_lock"]["evaluator_fingerprint"] = "replacement-fingerprint"
    case_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_suite(
        suite_root=suite_root,
        output_dir=output_dir,
        model_root=model_root,
        runner=runner,
        resume=True,
        log=_quiet,
    )

    assert len(runner.calls) == 2
    rows = read_rows(output_dir / PER_CASE_RESULTS_FILENAME)
    assert len(rows) == 1
    assert rows[0]["evaluator_lock"]["evaluator_fingerprint"] == (
        "replacement-fingerprint"
    )


def test_full_suite_manifest_runs_a_dimension_subset(tmp_path: Path) -> None:
    suite_root = build_default_suite_dir(tmp_path / "suite")
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_a/main", "scene_b/first"])
    manifest = tmp_path / "candidates.jsonl"
    rows = [
        {"model": "genie3", "case_id": case_id, "seed": "0001",
         "candidate_video": str(model_root / case_id / "0001.mp4")}
        for case_id in ("scene_a/main", "scene_b/first")
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = run_suite(
        suite_root=suite_root,
        output_dir=tmp_path / "out",
        candidate_manifest=manifest,
        dimensions=["toy_metric"],
        dry_run=True,
        log=_quiet,
    )
    assert [item["case_id"] for item in result["work_items"]] == ["scene_a/main"]

    # Case ids unknown to the suite are rejected.
    rows.append({**rows[0], "case_id": "scene_z/typo"})
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported official Lite case_id 'scene_z/typo'"):
        run_suite(
            suite_root=suite_root,
            output_dir=tmp_path / "out",
            candidate_manifest=manifest,
            dimensions=["toy_metric"],
            dry_run=True,
            log=_quiet,
        )


def test_run_suite_dry_run_writes_nothing(tmp_path: Path) -> None:
    suite_root = build_default_suite_dir(tmp_path / "suite")
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_a/main"])

    result = run_suite(
        suite_root=suite_root,
        output_dir=tmp_path / "out",
        model_root=model_root,
        dry_run=True,
        log=_quiet,
    )

    assert result["dry_run"] is True
    assert result["num_work_items"] == 1
    item = result["work_items"][0]
    assert item["candidate_exists"] is True
    assert item["reference_exists"] is True
    assert not (tmp_path / "out").exists()


def test_run_suite_rejects_unknown_dimension(tmp_path: Path) -> None:
    suite_root = build_default_suite_dir(tmp_path / "suite")
    with pytest.raises(SuiteRunError, match="unknown dimension"):
        run_suite(
            suite_root=suite_root,
            output_dir=tmp_path / "out",
            model_root=tmp_path / "genie3",
            dimensions=["nope"],
            log=_quiet,
        )


def test_run_suite_requires_exactly_one_candidate_source(tmp_path: Path) -> None:
    suite_root = build_default_suite_dir(tmp_path / "suite")
    with pytest.raises(SuiteRunError, match="exactly one of"):
        run_suite(
            suite_root=suite_root,
            output_dir=tmp_path / "out",
            log=_quiet,
        )


def test_run_suite_model_yaml_params_apply_to_directory_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_root = _registry_only_suite(tmp_path)
    model_root = tmp_path / "genie3"
    write_candidates(model_root, ["scene_a/main"])
    (model_root / "model.yaml").write_text("max_pairs: 2\n", encoding="utf-8")
    mock_video_probing(monkeypatch)
    runner = RecordingRunner()

    run_suite(
        suite_root=suite_root,
        output_dir=tmp_path / "out",
        model_root=model_root,
        runner=runner,
        log=_quiet,
    )

    assert runner.calls[0]["max_pairs"] == 2
