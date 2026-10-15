from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from evaluation.lite.action_support import declared_target, unsupported_reason
from evaluation.lite.output import PER_CASE_RESULTS_FILENAME, read_rows
from evaluation.lite.suite import run_suite
from lite_suite_test_utils import (
    DEFAULT_SUITE_YAML,
    mock_video_probing,
    regenerate_checksums,
    write_candidates,
    write_reference_case,
    write_suite,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": {"lpips": {"lpips_mean": 0.3, "lpips_median": 0.25}}}


def _quiet(_message: str) -> None:
    pass


def _declare_target(case_dir: Path, target: str) -> None:
    artifacts = case_dir / "reference" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "actions.json").write_text(json.dumps({"target": target}), encoding="utf-8")
    regenerate_checksums(case_dir)


def test_unsupported_reason_follows_the_backend_registry(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    assert declared_target(artifacts) is None
    assert unsupported_reason(artifacts, "waypoint15") is None  # no declaration: evaluated as before

    (artifacts / "actions.json").write_text(json.dumps({"target": "ball"}), encoding="utf-8")
    assert declared_target(artifacts) == "ball"
    reason = unsupported_reason(artifacts, "waypoint15")
    assert reason is not None and "ball" in reason and "no object channel" in reason
    assert "cannot move the ball" in (unsupported_reason(artifacts, "lingbot_world_cam") or "")
    for supported in ("unity", "project_genie", "lingbot_world_act"):
        assert unsupported_reason(artifacts, supported) is None
    assert unsupported_reason(artifacts, "some_unregistered_model") is None


def test_registered_real_frame_rate_is_applied_without_a_model_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = copy.deepcopy(DEFAULT_SUITE_YAML)
    del payload["dimensions"]["toy_promptcons"]
    suite_root = write_suite(tmp_path / "suite", payload)
    write_reference_case(suite_root / "cases")
    mock_video_probing(monkeypatch)

    seen: dict[str, object] = {}
    for model in ("lingbot_world_act", "project_genie"):
        runner = RecordingRunner()
        model_root = tmp_path / model
        write_candidates(model_root, ["scene_a/main"])
        run_suite(
            suite_root=suite_root, output_dir=tmp_path / f"out_{model}", model_root=model_root,
            runner=runner, log=_quiet,
        )
        seen[model] = runner.calls[0].get("candidate_fps_override")

    assert seen["lingbot_world_act"] == 10.0  # one frame per 100 ms action tick
    assert seen["project_genie"] is None  # no registered real frame rate

    # An explicit model.yaml value still wins over the registry.
    model_root = tmp_path / "lingbot_world_act"
    (model_root / "model.yaml").write_text("candidate_fps_override: 12.5\n", encoding="utf-8")
    runner = RecordingRunner()
    run_suite(
        suite_root=suite_root, output_dir=tmp_path / "out_explicit", model_root=model_root,
        runner=runner, log=_quiet,
    )
    assert runner.calls[0]["candidate_fps_override"] == 12.5


def test_run_suite_reports_unsupported_instead_of_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = copy.deepcopy(DEFAULT_SUITE_YAML)
    del payload["dimensions"]["toy_promptcons"]
    suite_root = write_suite(tmp_path / "suite", payload)
    case_dir = write_reference_case(suite_root / "cases")
    _declare_target(case_dir, "ball")
    mock_video_probing(monkeypatch)
    runner = RecordingRunner()

    results = {}
    for model in ("waypoint15", "project_genie"):
        model_root = tmp_path / model
        write_candidates(model_root, ["scene_a/main"])
        output = tmp_path / f"out_{model}"
        aggregate = run_suite(
            suite_root=suite_root, output_dir=output, model_root=model_root, runner=runner, log=_quiet
        )
        results[model] = (output, aggregate)

    wp_output, wp_aggregate = results["waypoint15"]
    row = read_rows(wp_output / PER_CASE_RESULTS_FILENAME)[0]
    assert row["status"] == "unsupported"
    assert "no object channel" in row["error"]
    summary = wp_aggregate["models"]["waypoint15"]["dimensions"]["toy_metric"]
    assert summary["num_completed"] == 0
    assert summary["num_unsupported"] == 1
    assert summary["headline"]["value"] is None
    with (wp_output / "aggregate_summary.csv").open(encoding="utf-8") as handle:
        cells = list(csv.DictReader(handle))[0]
    assert cells["toy_metric:lpips_mean"] == "unsupported"

    gn_output, gn_aggregate = results["project_genie"]
    assert read_rows(gn_output / PER_CASE_RESULTS_FILENAME)[0]["status"] == "completed"
    assert gn_aggregate["models"]["project_genie"]["dimensions"]["toy_metric"]["num_unsupported"] == 0
    assert len(runner.calls) == 1  # only the supported model reached the metric runner
