from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import tools.run_prepared_table_evaluations as module
from tools.run_prepared_table_evaluations import BLOCKED, EXCLUDED, PLANS, _result_errors


def test_automation_plan_covers_all_table_cells_once() -> None:
    case_ids = [plan.case_id for plan in PLANS] + list(BLOCKED) + list(EXCLUDED)
    assert len(case_ids) == 16
    assert len(set(case_ids)) == 16


def test_racing_third_uses_matched_dynamic_unity_ground_truth() -> None:
    plan = next(item for item in PLANS if item.case_id == "table4/RacingCar/third")
    assert plan.preparation == "racing_unity_capture"
    params = module._racing_unity_params(plan)
    assert params["kin_text_prompt"] == "car."
    assert "unity_scene_runs/captures/table4__RacingCar__third" in params[
        "kin_unity_object_transforms_json"
    ]
    assert params["kin_mechanics_estimator"] == "aligned_peak_finite_difference"
    assert params["kin_event_alignment_tolerance_frames"] == 5
    masses = json.loads(
        open(params["kin_unity_object_masses_json"], encoding="utf-8").read()
    )
    assert masses == {"car": 1000.0}


def test_billiard_table7_uses_matching_unity_capture_artifacts() -> None:
    plan = next(
        item for item in PLANS if item.case_id == "table7/BilliardGame/perturb_in"
    )
    params, reference = module._billiard_unity_inputs(plan)
    assert reference.is_file()
    assert "table7__BilliardGame__perturb_in" in str(reference)
    assert Path(params["eoc_unity_object_transforms_json"]).is_file()
    assert Path(params["eoc_unity_poses_npy"]).is_file()
    assert Path(params["eoc_unity_intrinsics_npy"]).is_file()
    assert params["eoc_text_prompt"] == "yellow ball.|red ball."
    assert params["eoc_character_prompt"] == "white ball."


def test_billiard_table4_keeps_first_person_ball_control_only() -> None:
    plan = next(
        item for item in PLANS if item.case_id == "table4/BilliardGame/first"
    )
    params, reference = module._billiard_kinematics_inputs(plan)
    assert reference.is_file()
    assert "WASD directly controls the white ball" in params["text_prompt"]
    assert params["kin_text_prompt"] == "yellow ball."
    assert params["kin_track_point"] == "bbox_bottom_center"
    assert params["kin_mechanics_estimator"] == "aligned_peak_finite_difference"
    assert params["kin_event_alignment_tolerance_frames"] == 5
    assert Path(params["kin_plane_calibration_json"]).is_file()
    assert "table4__BilliardGame__first" in str(reference)
    assert "table4/BilliardGame/third" in EXCLUDED
    assert "table4/RacingCar/first" in EXCLUDED


def test_camera_plans_encode_conditioning_orientation_and_action_direction() -> None:
    plans = {plan.case_id: plan for plan in PLANS}
    assert plans["table7/WanderingGarden/360_in"].reference_start_frame == 0
    assert plans["table7/WanderingGarden/360_in"].target_yaw_deg == -360.0
    assert plans["table7/WanderingGarden/360_out"].reference_start_frame == 75
    assert plans["table7/WanderingGarden/360_out"].target_yaw_deg == 360.0
    assert plans["table7/MirrorRollingBall/perturb_out"].reference_start_frame == 135
    assert plans["table7/MirrorRollingBall/perturb_out"].target_yaw_deg == 45.0


def test_result_validation_rejects_zero_exit_error_rows(tmp_path) -> None:
    plan = PLANS[0]
    path = tmp_path / "per_case_results.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "dimension": plan.dimension,
                    "case_id": plan.lite_case_id,
                    "model": "project_genie",
                    "seed": f"run{index}",
                    "status": "error" if index == 2 else "completed",
                    "error": "device failed" if index == 2 else None,
                }
            )
            + "\n"
            for index in range(5)
        )
    )
    assert _result_errors(path, plan) == ["run2: error: device failed"]


def test_scoped_comparison_marks_reconstructed_rows(
    tmp_path, monkeypatch
) -> None:
    comparison = {
        "complete": True,
        "rows": [
            {"status": "complete", "table": 4},
            {"status": "pending", "table": 7},
        ],
    }

    def fake_write(document, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "comparison.md").write_text(
            "# Comparison\n\nFive-run sample statistics (`ddof=1`).\n"
        )

    monkeypatch.setattr(module, "write_comparison", fake_write)
    module._write_scoped_comparison(
        comparison,
        tmp_path,
        claim_scope="paper_protocol_reconstruction_not_exact_paper_parity",
    )

    assert comparison["rows"][0]["status"] == "complete_reconstructed"
    assert comparison["rows"][1]["status"] == "pending"
    assert "Claim scope: `paper_protocol_reconstruction_not_exact_paper_parity`" in (
        tmp_path / "comparison.md"
    ).read_text()


def test_consolidation_selects_latest_complete_attempt_and_ignores_incomplete(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "load_case",
        lambda _path: SimpleNamespace(
            evaluator_lock={"evaluator_fingerprint": "current"}
        ),
    )
    root = tmp_path / "reports"

    def write(name: str, *, incomplete: list[str], value: float) -> Path:
        path = root / name / "five_run_reconstructed_pose_corrected_cpu" / "rerun_results.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "provenance": {
                        "source_case_id": "Scene/source",
                        "target_case_id": "table7/Scene/case",
                        "incomplete_metrics": incomplete,
                        "evaluation_settings": {
                            "evaluator_lock": {
                                "evaluator_fingerprint": "current"
                            }
                        },
                    },
                    "value": value,
                }
            )
        )
        return path

    older = write("older", incomplete=[], value=1.0)
    newer = write("newer", incomplete=[], value=2.0)
    incomplete = write("incomplete", incomplete=["run2:dist_mae_pct"], value=3.0)
    older.touch()
    newer.touch()
    incomplete.touch()
    # Force deterministic ordering independent of filesystem timestamp resolution.
    import os

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    os.utime(incomplete, (3, 3))

    selected, ignored = module._select_consolidated_sources(root)

    assert [(path, document["value"]) for path, document in selected] == [(newer, 2.0)]
    assert ignored["incomplete"][0]["path"] == str(incomplete.resolve())
    assert ignored["superseded"][0]["path"] == str(older.resolve())


def test_selected_excluded_cell_is_not_an_incomplete_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "_write_automation_status",
        lambda **kwargs: {
            "status": kwargs["status"],
            "cases": kwargs["rows"],
        },
    )

    report, exit_code = module.run(
        selected={"table4/RacingCar/first"},
        device="cpu",
        prepare_only=True,
        fail_incomplete=True,
    )

    assert exit_code == 0
    assert report["status"] == "complete"
    assert report["cases"] == [
        {
            "case_id": "table4/RacingCar/first",
            "status": "excluded_by_user",
            "reason": EXCLUDED["table4/RacingCar/first"],
        }
    ]


def test_fail_incomplete_rejects_null_required_metric(tmp_path, monkeypatch) -> None:
    plan = next(item for item in PLANS if item.case_id.startswith("table7/"))
    run_root = tmp_path / "run-root"
    run_root.mkdir()
    manifest = tmp_path / "candidate_manifest.jsonl"
    manifest.write_text("", encoding="utf-8")

    monkeypatch.setattr(module, "_run_root", lambda _case_id: run_root)
    monkeypatch.setattr(module, "_report_root", lambda _case_id: tmp_path / "reports")
    monkeypatch.setattr(
        module,
        "build_summary",
        lambda _root, _repeats: {"status": "validated", "errors": []},
    )
    monkeypatch.setattr(module, "write_summary", lambda _summary, _prefix: None)
    monkeypatch.setattr(
        module,
        "_prepare",
        lambda _plan, _root, _summary, _output: {
            "candidate_manifest": str(manifest)
        },
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"returncode": 0})(),
    )
    monkeypatch.setattr(module, "_result_errors", lambda _path, _plan: [])
    monkeypatch.setattr(
        module,
        "_write_comparison_artifacts",
        lambda _plan, _prepared, _report: {
            "incomplete_metrics": ["run1:dist_mae_pct"]
        },
    )
    monkeypatch.setattr(module, "refresh_consolidated_comparison", lambda: {})
    monkeypatch.setattr(
        module,
        "_write_automation_status",
        lambda **kwargs: {
            "status": kwargs["status"],
            "cases": kwargs["rows"],
        },
    )

    report, exit_code = module.run(
        selected={plan.case_id},
        device="cpu",
        prepare_only=False,
        fail_incomplete=True,
    )

    assert exit_code == 2
    assert report["status"] == "incomplete"
    assert report["cases"][0]["status"] == "results_complete_metrics_incomplete"
    assert report["cases"][0]["incomplete_metrics"] == ["run1:dist_mae_pct"]
    assert report["cases"][0]["regeneration_required"][0]["repeat_id"] == "run1"
