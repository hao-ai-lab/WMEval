from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "audit_table_reproduction.py"
SPEC = importlib.util.spec_from_file_location("audit_table_reproduction", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_queue_requires_calibration_for_every_table7_case() -> None:
    calibrated = "\n".join(
        f"runner --case table7/Scene/case{index} --turn-calibration case=period.json"
        for index in range(12)
    )
    assert MODULE._queue_uses_calibrated_fixed_turns(calibrated) is True
    assert (
        MODULE._queue_uses_calibrated_fixed_turns(
            calibrated.replace(" --turn-calibration case=period.json", "", 1)
        )
        is False
    )
    assert (
        MODULE._queue_uses_calibrated_fixed_turns(
            calibrated + " --closed-loop-yaw"
        )
        is False
    )


def test_matching_result_rows_requires_current_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "per_case_results.jsonl"
    rows = [
        {
            "case_id": "Scene/case",
            "dimension": "dimension",
            "model": "project_genie",
            "seed": "run0",
            "status": "completed",
            "evaluator_lock": {"evaluator_fingerprint": "current"},
        },
        {
            "case_id": "Scene/case",
            "dimension": "dimension",
            "model": "project_genie",
            "seed": "run1",
            "status": "completed",
            "evaluator_lock": {"evaluator_fingerprint": "stale"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    selected = MODULE._matching_result_rows(
        path,
        lite_case_id="Scene/case",
        dimension="dimension",
        evaluator_fingerprint="current",
    )
    assert set(selected) == {"run0"}


def test_generation_status_distinguishes_validated_and_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    assets = repo / "runs" / "tables_4_7" / "prepared_assets"
    assets.mkdir(parents=True)
    (assets / "asset_manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "table7/Scene/ready",
                        "status": "prepared",
                        "view": "first_person",
                        "environment_prompt": "A scene",
                        "character_prompt": "",
                        "prompt_provenance_status": "reconstructed",
                        "conditioning": {"sha256": "frame-hash"},
                        "actions": {
                            "trajectory_stem": "turn",
                            "files": {
                                "turn.actions.jsonl": {"sha256": "actions-hash"}
                            },
                        },
                    },
                    {
                        "case_id": "table7/Scene/blocked",
                        "status": "blocked",
                        "issues": ["missing"],
                    },
                ]
            }
        )
    )
    validation = repo / "runs" / "project_genie" / "table7/Scene/ready"
    validation.mkdir(parents=True)
    (validation / "five_run_validation.json").write_text(
        json.dumps(
            {
                "status": "validated",
                "generation_count": 5,
                "repeat_ids": [f"run{index}" for index in range(5)],
                "errors": [],
            }
        )
    )
    for index in range(5):
        run = validation / f"run{index}"
        run.mkdir()
        artifact = run / "observed.mp4"
        artifact.write_bytes(f"run{index}".encode())
        (run / "receipt.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "generation_mode": "new_world_generation",
                    "table_repeat_eligible": True,
                    "repeat_id": f"run{index}",
                    "view": "first_person",
                    "environment_prompt": "A scene",
                    "character_prompt": "",
                    "prompt_provenance_status": "reconstructed",
                    "first_frame_sha256": "frame-hash",
                    "actions_sha256": "actions-hash",
                    "trajectory_validation_status": "calibrated_backend_adapter",
                    "evaluation_artifact_status": "action_aligned_capture_ready",
                    "observed_video": str(artifact),
                    "observed_video_sha256": hashlib.sha256(
                        artifact.read_bytes()
                    ).hexdigest(),
                }
            )
        )
    monkeypatch.setattr(MODULE, "REPO_ROOT", repo)
    monkeypatch.setattr(MODULE, "ASSET_ROOT", assets)
    result = MODULE._generation_status()
    assert result["validated_case_count"] == 1
    assert result["blocked_case_count"] == 1
    assert result["invalid_or_missing_case_count"] == 0


def test_generation_status_rejects_receipt_from_stale_conditioning(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    assets = repo / "runs/tables_4_7/prepared_assets"
    assets.mkdir(parents=True)
    case = {
        "case_id": "table4/RacingCar/third",
        "status": "prepared",
        "view": "third_person",
        "environment_prompt": "Road",
        "character_prompt": "Car",
        "prompt_provenance_status": "reconstructed",
        "conditioning": {"sha256": "new-frame"},
        "actions": {
            "trajectory_stem": "drive",
            "files": {"drive.actions.jsonl": {"sha256": "actions"}},
        },
    }
    (assets / "asset_manifest.json").write_text(json.dumps({"cases": [case]}))
    root = repo / "runs/project_genie/table4/RacingCar/third"
    root.mkdir(parents=True)
    (root / "five_run_validation.json").write_text(
        json.dumps(
            {
                "status": "validated",
                "generation_count": 5,
                "repeat_ids": [f"run{index}" for index in range(5)],
                "errors": [],
            }
        )
    )
    for index in range(5):
        run = root / f"run{index}"
        run.mkdir()
        artifact = run / "observed.mp4"
        artifact.write_bytes(b"video")
        (run / "receipt.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "generation_mode": "new_world_generation",
                    "table_repeat_eligible": True,
                    "repeat_id": f"run{index}",
                    "view": "third_person",
                    "environment_prompt": "Road",
                    "character_prompt": "Car",
                    "prompt_provenance_status": "reconstructed",
                    "first_frame_sha256": "old-frame",
                    "actions_sha256": "actions",
                    "trajectory_validation_status": "calibrated_backend_adapter",
                    "evaluation_artifact_status": "action_aligned_capture_ready",
                    "observed_video": str(artifact),
                    "observed_video_sha256": hashlib.sha256(
                        artifact.read_bytes()
                    ).hexdigest(),
                }
            )
        )
    monkeypatch.setattr(MODULE, "REPO_ROOT", repo)
    monkeypatch.setattr(MODULE, "ASSET_ROOT", assets)

    result = MODULE._generation_status()

    assert result["validated_case_count"] == 0
    assert result["invalid_or_missing_case_count"] == 1
    assert {
        row["status"] for row in result["cases"][0]["receipts"]
    } == {"input_mismatch_complete"}


def test_table7_result_artifacts_require_current_fingerprint(tmp_path: Path) -> None:
    report = tmp_path / "report"
    comparison = report / "comparison"
    comparison.mkdir(parents=True)
    results = report / "per_case_results.jsonl"
    results.write_text("")
    (comparison / "comparison.json").write_text("{}")
    (comparison / "comparison.md").write_text("# comparison\n")
    rerun = {
        "repeat_ids": [f"run{index}" for index in range(5)],
        "provenance": {
            "target_case_id": "table7/Scene/case",
            "incomplete_metrics": [],
            "evaluation_settings": {
                "evaluator_lock": {"evaluator_fingerprint": "current"}
            },
        },
    }
    (report / "rerun_results.json").write_text(json.dumps(rerun))

    valid, details = MODULE._result_artifacts_valid(
        case_id="table7/Scene/case",
        results_path=results,
        evaluator_fingerprint="current",
    )
    assert valid is True
    assert details["artifact_status"] == "validated"

    stale, details = MODULE._result_artifacts_valid(
        case_id="table7/Scene/case",
        results_path=results,
        evaluator_fingerprint="new",
    )
    assert stale is False
    assert details["artifact_status"] == "invalid"

    rerun["provenance"]["incomplete_metrics"] = ["run1:dist_mae_pct"]
    (report / "rerun_results.json").write_text(json.dumps(rerun))
    incomplete, details = MODULE._result_artifacts_valid(
        case_id="table7/Scene/case",
        results_path=results,
        evaluator_fingerprint="current",
    )
    assert incomplete is False
    assert details["artifact_status"] == "incomplete_metrics"
    assert details["incomplete_metrics"] == ["run1:dist_mae_pct"]


def test_evaluation_status_rejects_results_from_stale_generation(monkeypatch) -> None:
    plan = MODULE.PLANS[0]
    monkeypatch.setattr(MODULE, "PLANS", (plan,))
    monkeypatch.setattr(MODULE, "BLOCKED", {})
    monkeypatch.setattr(MODULE, "EXCLUDED", {})
    result = MODULE._evaluation_status(
        {
            "cases": [
                {
                    "case_id": plan.case_id,
                    "status": "invalid",
                }
            ]
        }
    )
    assert result["completed_case_count"] == 0
    assert result["in_progress_case_count"] == 1
    assert result["cases"][0]["status"] == "stale_or_incomplete_generation_inputs"


def test_evaluation_status_does_not_count_user_exclusion_as_runnable(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "PLANS", ())
    monkeypatch.setattr(MODULE, "BLOCKED", {})
    monkeypatch.setattr(MODULE, "EXCLUDED", {"table4/BilliardGame/third": "first only"})
    result = MODULE._evaluation_status()
    assert result["runnable_case_count"] == 0
    assert result["blocked_case_count"] == 0
    assert result["excluded_case_count"] == 1
    assert result["cases"][0]["status"] == "excluded_by_user"


def test_metric_source_readiness_fails_closed_for_single_mirror_object(
    tmp_path: Path, monkeypatch,
) -> None:
    mirror = (
        tmp_path
        / "eval_sets/wmeval_lite_v1/cases/MirrorRollingBall/prompt_third"
        / "reference/artifacts"
    )
    mirror.mkdir(parents=True)
    (mirror / "object_transforms.json").write_text(
        json.dumps({"objects": {"yellow sphere": {"positions_world": []}}})
    )
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path)

    result = MODULE._metric_source_readiness()

    assert result["status"] == "blocked"
    metric = result["metrics"][0]
    assert metric["status"] == "blocked_insufficient_unique_unity_objects"
    assert metric["unique_object_count"] == 1
    assert metric["minimum_required"] == 2


def test_live_browser_receipt_proves_end_to_end_artifacts(
    tmp_path: Path, monkeypatch,
) -> None:
    run = tmp_path / "runs/project_genie/table7/Scene/variant/run1"
    run.mkdir(parents=True)
    artifacts = {}
    for name in ("frame.png", "actions.jsonl", "trace.json", "video.mp4", "alignment.json"):
        path = run / name
        path.write_bytes(name.encode())
        artifacts[name] = (str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    receipt = {
        "status": "complete",
        "generation_mode": "new_world_generation",
        "table_repeat_eligible": True,
        "first_frame": artifacts["frame.png"][0],
        "first_frame_sha256": artifacts["frame.png"][1],
        "actions": artifacts["actions.jsonl"][0],
        "actions_sha256": artifacts["actions.jsonl"][1],
        "action_trace": artifacts["trace.json"][0],
        "action_trace_sha256": artifacts["trace.json"][1],
        "evaluation_video": artifacts["video.mp4"][0],
        "evaluation_video_sha256": artifacts["video.mp4"][1],
        "native_alignment_report": artifacts["alignment.json"][0],
        "native_alignment_report_sha256": artifacts["alignment.json"][1],
        "native_video_timeline_alignment": "validated_visual_correspondence",
        "environment_prompt": "A scene",
        "character_prompt": "",
        "view": "first_person",
        "screenshots": {
            key: {} for key in (
                "frame_uploaded", "prompts_ready", "generation_requested",
                "world_ready", "actions_complete", "complete",
            )
        },
        "upload_policy_acceptances": [{
            "button_label": "I agree",
            "policy_label": "Prohibited Use Policy",
            "mode": "automated_explicit_cli_opt_in",
        }],
    }
    receipt_path = run / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path)

    result = MODULE._validate_live_browser_receipt(receipt_path)
    assert result["status"] == "validated"
    assert result["case_id"] == "table7/Scene/variant"
    assert result["repeat_id"] == "run1"

    (run / "video.mp4").write_bytes(b"tampered")
    assert MODULE._validate_live_browser_receipt(receipt_path)["status"] == "invalid"
