from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import cv2


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_table_assets.py"
SPEC = importlib.util.spec_from_file_location("prepare_table_assets", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_direct_conditioning_source_enforces_pinned_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"not-the-pinned-frame")

    with pytest.raises(ValueError, match="does not match pinned value"):
        MODULE._materialize_frame(
            tmp_path,
            {
                "kind": "direct",
                "path": "source.png",
                "expected_sha256": "0" * 64,
            },
            tmp_path / "output.png",
        )


def test_conditioning_source_rejects_images_below_genie_minimum(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.zeros((512, 512, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="must be at least 1280x704"):
        MODULE._materialize_frame(
            tmp_path,
            {"kind": "direct", "path": "source.png"},
            tmp_path / "output.png",
        )


def test_orientation_validation_accepts_opposing_forward_axes(tmp_path: Path) -> None:
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, 2] *= -1
    np.save(tmp_path / "poses.npy", poses)
    result = MODULE._validate_orientation(
        tmp_path,
        {
            "orientation_validation": {
                "poses": "poses.npy",
                "base_frame": 0,
                "derived_frame": 1,
                "maximum_forward_dot": -0.99,
            }
        },
    )
    assert result is not None
    assert result["horizontal_forward_dot"] == -1.0


def test_prepare_materializes_all_cases_and_fail_closes_missing_views(tmp_path: Path) -> None:
    report = MODULE.prepare(MODULE.TABLE_ROOT, tmp_path)
    assert report["case_count"] == 16
    assert report["repeat_count"] == 80
    assert report["evaluation_status_counts"] == {
        "blocked_paper_metric_parity": 20,
        "excluded": 10,
        "runnable_diagnostic_not_paper_comparable": 40,
        "runnable_metric_reconstruction": 10,
    }
    cases = {row["case_id"]: row for row in report["cases"]}
    assert cases["table7/MirrorRollingBall/perturb_out"]["status"] == "prepared_with_caveats"
    mirror = cases["table7/MirrorRollingBall/perturb_out"]["conditioning"]
    assert mirror["orientation_validation"]["horizontal_forward_dot"] < -0.95
    assert Path(mirror["path"]).is_file()
    assert cases["table4/BilliardGame/third"]["status"] == "excluded"
    assert "first-person ball-level view" in cases["table4/BilliardGame/first"][
        "environment_prompt"
    ]
    assert cases["table7/BilliardGame/360_in"]["status"] == "prepared_with_caveats"
    queue = [json.loads(line) for line in (tmp_path / "generation_queue.jsonl").read_text().splitlines()]
    assert len(queue) == 80
    assert sum(row["status"] == "ready" for row in queue) == 70
    assert sum(row["status"] == "excluded" for row in queue) == 10
    evaluation_queue = [
        json.loads(line)
        for line in (tmp_path / "evaluation_queue.jsonl").read_text().splitlines()
    ]
    assert len(evaluation_queue) == 80
    assert sum(row["status"] == "runnable_metric_reconstruction" for row in evaluation_queue) == 10
    billiard_first = next(
        row
        for row in evaluation_queue
        if row["case_id"] == "table4/BilliardGame/first"
    )
    assert billiard_first["status"] == "runnable_metric_reconstruction"
    assert (
        billiard_first["evaluator_status"]
        == "runnable_metric_reconstruction"
    )
    racing_first = next(
        row
        for row in evaluation_queue
        if row["case_id"] == "table4/RacingCar/first"
    )
    assert racing_first["status"] == "excluded"
    assert cases["table4/RacingCar/first"]["status"] == "excluded"
    assert (
        cases["table4/RacingCar/third"]["evaluation"]["status"]
        == "runnable_metric_reconstruction"
    )
    assert "mechanical_exact_match_error_rate_unity_mass_metric_trajectory" in cases[
        "table4/RacingCar/third"
    ]["evaluation"]["issues"]
    runner = tmp_path / "run_ready_project_genie.sh"
    assert runner.stat().st_mode & 0o111
    runner_text = runner.read_text()
    assert "project_genie_agent.py launch" in runner_text
    assert "failures=$((failures + 1))" in runner_text
    assert "table4/RacingCar/first" not in runner_text
    racing_line = next(
        line for line in runner_text.splitlines() if "table4/RacingCar/third" in line
    )
    garden_360_line = next(
        line for line in runner_text.splitlines() if "table7/WanderingGarden/360_in" in line
    )
    assert "--allow-missing-native" in racing_line
    assert "--archive-stale-inputs" in racing_line
    assert "--download-timeout-ms 20000" in racing_line
    assert "--allow-missing-native" in garden_360_line
    assert "--turn-calibration" in garden_360_line
    assert "--closed-loop-yaw" not in garden_360_line
    full_runner = tmp_path / "run_full_pipeline.sh"
    assert full_runner.stat().st_mode & 0o111
    assert "run_prepared_table_evaluations.py" in full_runner.read_text()
    assert "--fail-incomplete" in full_runner.read_text()
    readme = (tmp_path / "README.md").read_text()
    assert "| Case | Generation | Evaluation |" in readme
    assert "10 runnable metric-reconstruction repeats" in readme
