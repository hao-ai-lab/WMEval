"""Release gates on Lite rows and aggregates: input contract, per-metric status,
action-support status, the suite repeat-policy coverage gate, and the declared
action target fallback (table_cell.json / trajectory.yaml)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.lite.action_support import declared_target, support_record
from evaluation.lite.output import PER_CASE_RESULTS_FILENAME, read_rows
from evaluation.lite.suite import input_contract, metric_status, run_suite
from lite_suite_test_utils import (
    DEFAULT_SUITE_YAML,
    mock_video_probing,
    write_candidates,
    write_reference_case,
    write_suite,
)

SUITE_ROOT = Path(__file__).resolve().parents[1] / "lite"


def _quiet(_message: str) -> None:
    return None


class FixedRunner:
    def __call__(self, **kwargs):
        return {"results": {"lpips": {"lpips_mean": 0.3, "lpips_median": 0.25}}}


def _meta(fps, width, height, frames):
    return SimpleNamespace(fps=fps, width=width, height=height, frame_count=frames, duration_sec=frames / fps)


def test_input_contract_flags_mismatches_only_for_unnormalized_cases() -> None:
    validated = SimpleNamespace(candidate=_meta(16, 832, 480, 80), reference=_meta(30, 1280, 720, 300))
    report = input_contract(validated, normalization="none")
    assert report["within_spec"] is False
    assert report["flags"] == {
        "fps_mismatch": True, "resolution_mismatch": True, "aspect_mismatch": True, "duration_mismatch": True,
    }
    assert input_contract(validated, normalization="match_reference")["within_spec"] is None
    same = SimpleNamespace(candidate=_meta(30, 1280, 720, 300), reference=_meta(30, 1280, 720, 300))
    assert input_contract(same, normalization="none")["within_spec"] is True


def test_metric_status_marks_non_finite_values_unmeasurable() -> None:
    status = metric_status("completed", {"a": 0.5, "b": float("nan"), "label": "x"})
    assert status == {"a": "ok", "b": "unmeasurable"}
    assert metric_status("unsupported", {"a": 0.5}) == {"a": "unsupported"}


@pytest.mark.integration("case_data")
def test_declared_target_falls_back_to_bundled_trajectory() -> None:
    cases = SUITE_ROOT / "cases"
    assert declared_target(cases / "RacingCar/kinematics_third/reference/artifacts") == "ego_vehicle"
    assert declared_target(cases / "WanderingGarden/scene_360_in/reference/artifacts") == "camera"
    assert declared_target(cases / "ChessBoard/walk_third/reference/artifacts") == "character"
    assert declared_target(cases / "LiquidContainer/fluid_rest/reference/artifacts") == "liquid_container"
    record = support_record(cases / "BilliardGame/kinematics_first/reference/artifacts", "matrix_game_2")
    assert record == {"target": "cue_ball", "status": "target_entity_validation_required"}
    assert support_record(cases / "WanderingGarden/scene_360_in/reference/artifacts", "hy_worldplay") == {
        "target": "camera", "status": "trajectory_scoped",
    }


def test_coverage_gate_withholds_headline_until_required_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = copy.deepcopy(DEFAULT_SUITE_YAML)
    del payload["dimensions"]["toy_promptcons"]
    payload["repeat_policy"] = {"required_repeats": 2}
    suite_root = write_suite(tmp_path / "suite", payload)
    write_reference_case(suite_root / "cases")
    mock_video_probing(monkeypatch)

    model_root = tmp_path / "model"
    write_candidates(model_root, ["scene_a/main"])
    output = tmp_path / "out"
    aggregate = run_suite(suite_root=suite_root, output_dir=output, model_root=model_root,
                          runner=FixedRunner(), log=_quiet)
    summary = next(iter(aggregate["models"].values()))["dimensions"]["toy_metric"]
    assert summary["coverage"]["complete"] is False
    assert summary["coverage"]["cases_missing_repeats"] == ["scene_a/main"]
    assert summary["headline"]["value"] is None
    assert summary["headline"]["provisional_value"] == pytest.approx(0.3)

    row = read_rows(output / PER_CASE_RESULTS_FILENAME)[0]
    assert row["metric_status"]["lpips_mean"] == "ok"
    assert "input_contract" in row
