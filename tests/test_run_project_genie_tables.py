from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "run_project_genie_tables.py"
SPEC = importlib.util.spec_from_file_location("run_project_genie_tables", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.integration("table_runs")
def test_full_plan_has_five_runs_per_table_cell(tmp_path: Path) -> None:
    tasks = MODULE.task_plan(MODULE.TABLE_ROOT, tmp_path, set())
    assert len(tasks) == 70
    assert len({(task["case_id"], task["repeat_id"]) for task in tasks}) == 70
    assert not any(task["case_id"] == "table4/BilliardGame/third" for task in tasks)
    assert not any(task["case_id"] == "table4/RacingCar/first" for task in tasks)
    assert sum(task["status"] == "pending" for task in tasks) == 0
    assert sum(task["status"].startswith("blocked_") for task in tasks) == 70
    assert sum(task["status"] == "blocked_uncalibrated_adapter" for task in tasks) == 50


def test_plan_filters_exact_case(tmp_path: Path) -> None:
    tasks = MODULE.task_plan(
        MODULE.TABLE_ROOT, tmp_path, {"table7/MirrorRollingBall/360_in"}
    )
    assert len(tasks) == 5
    assert {task["repeat_id"] for task in tasks} == {"run0", "run1", "run2", "run3", "run4"}


def test_plan_filters_single_repeat(tmp_path: Path) -> None:
    tasks = MODULE.task_plan(
        MODULE.TABLE_ROOT,
        tmp_path,
        {"table7/MirrorRollingBall/360_in"},
        selected_repeats={"run2"},
    )
    assert len(tasks) == 1
    assert tasks[0]["repeat_id"] == "run2"


@pytest.mark.integration("table_runs")
def test_default_plan_uses_reconstructed_billiard_orientation_frames(
    tmp_path: Path,
) -> None:
    billiard = MODULE.task_plan(
        MODULE.TABLE_ROOT,
        tmp_path,
        {"table7/BilliardGame/perturb_in"},
    )
    assert all(
        "table7__BilliardGame__perturb_in" in task["first_frame"]
        for task in billiard
    )
    assert {task["status"] for task in billiard} == {
        "blocked_uncalibrated_adapter"
    }

    mirror = MODULE.task_plan(
        MODULE.TABLE_ROOT,
        tmp_path,
        {"table7/MirrorRollingBall/perturb_out"},
    )
    assert all("table7__MirrorRollingBall__perturb_out" in task["first_frame"] for task in mirror)
    assert {task["status"] for task in mirror} == {
        "blocked_unresolved_orientation_frame"
    }


def test_scene_first_frame_override_applies_to_all_scene_cases(tmp_path: Path) -> None:
    frame = tmp_path / "unity_frame0.png"
    frame.write_bytes(b"png")
    tasks = MODULE.task_plan(
        MODULE.TABLE_ROOT,
        tmp_path / "out",
        {"table7/WanderingGarden/360_in"},
        {"WanderingGarden": frame},
    )
    assert len(tasks) == 5
    assert {task["first_frame"] for task in tasks} == {str(frame.resolve())}
    assert {task["first_frame_source"] for task in tasks} == {"override"}


def test_case_override_unlocks_only_the_target_orientation(tmp_path: Path) -> None:
    frame = tmp_path / "outward.png"
    frame.write_bytes(b"png")
    calibration = tmp_path / "turn_calibration.json"
    calibration.write_text("{}")
    selected = {
        "table7/WanderingGarden/360_in",
        "table7/WanderingGarden/360_out",
    }
    tasks = MODULE.task_plan(
        MODULE.TABLE_ROOT,
        tmp_path / "out",
        selected,
        {"table7/WanderingGarden/360_out": frame},
        turn_calibration_overrides={"WanderingGarden": calibration},
    )
    statuses = {(task["case_id"], task["status"]) for task in tasks}
    assert statuses == {
        ("table7/WanderingGarden/360_in", "pending"),
        ("table7/WanderingGarden/360_out", "pending"),
    }
    sources = {task["case_id"]: task["first_frame_source"] for task in tasks}
    assert sources["table7/WanderingGarden/360_in"] == "manifest"
    assert sources["table7/WanderingGarden/360_out"] == "override"


def test_yaw_case_stays_blocked_without_calibration(tmp_path: Path) -> None:
    tasks = MODULE.task_plan(
        MODULE.TABLE_ROOT,
        tmp_path,
        {"table7/WanderingGarden/360_in"},
    )
    assert {task["status"] for task in tasks} == {"blocked_uncalibrated_adapter"}


@pytest.mark.integration("table_runs")
def test_explicit_nominal_adapter_opt_in_unlocks_reconstructed_timing(tmp_path: Path) -> None:
    tasks = MODULE.task_plan(
        MODULE.TABLE_ROOT,
        tmp_path,
        {"table4/RacingCar/third"},
        allow_nominal_adapter=True,
    )
    assert {task["status"] for task in tasks} == {"pending"}
    assert {task["adapter_claim_scope"] for task in tasks} == {
        "reconstructed_nominal_timing"
    }


def test_closed_loop_scope_takes_precedence_over_nominal_opt_in(tmp_path: Path) -> None:
    tasks = MODULE.task_plan(
        MODULE.TABLE_ROOT,
        tmp_path,
        {"table7/WanderingGarden/360_in"},
        closed_loop_yaw=True,
        allow_nominal_adapter=True,
    )
    assert {task["status"] for task in tasks} == {"pending"}
    assert {task["adapter_claim_scope"] for task in tasks} == {
        "calibrated_or_closed_loop"
    }


def test_table7_prompts_match_motion_and_orientation() -> None:
    tasks = MODULE.task_plan(
        MODULE.TABLE_ROOT,
        Path("/tmp/wmeval-test-output"),
        {
            "table7/WanderingGarden/perturb_in",
            "table7/WanderingGarden/360_out",
        },
    )
    prompts = {task["case_id"]: task["environment_prompt"] for task in tasks}
    assert "45-degree" in prompts["table7/WanderingGarden/perturb_in"]
    assert "inward-facing" in prompts["table7/WanderingGarden/perturb_in"]
    assert "full 360-degree" in prompts["table7/WanderingGarden/360_out"]
    assert "outward-facing" in prompts["table7/WanderingGarden/360_out"]


def test_completed_receipt_requires_hashed_evaluation_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "trajectory_native.mp4"
    artifact.write_bytes(b"native trajectory")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "trajectory_validation_status": "closed_loop_visual_phase",
                "evaluation_artifact_status": "native_action_aligned_ready",
                "evaluation_video": str(artifact),
                "evaluation_video_sha256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
                "native_video_timeline_alignment": "validated_visual_correspondence",
            }
        )
    )
    alignment = tmp_path / "native_alignment.json"
    alignment.write_text("{}")
    payload = json.loads(receipt.read_text())
    payload["native_alignment_report"] = str(alignment)
    payload["native_alignment_report_sha256"] = hashlib.sha256(
        alignment.read_bytes()
    ).hexdigest()
    receipt.write_text(json.dumps(payload))
    assert MODULE.completed_receipt_status(receipt) == ("complete", True)
    artifact.write_bytes(b"tampered")
    assert MODULE.completed_receipt_status(receipt) == (
        "invalid_evaluation_artifact",
        False,
    )


def test_completed_receipt_rejects_missing_evaluation_artifact(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "trajectory_validation_status": "calibrated_visual_periodicity",
                "evaluation_artifact_status": "not_ready",
            }
        )
    )
    assert MODULE.completed_receipt_status(receipt) == (
        "incomplete_evaluation_artifact",
        False,
    )


def test_completed_receipt_rejects_changed_inputs(tmp_path: Path) -> None:
    artifact = tmp_path / "observed.mp4"
    artifact.write_bytes(b"capture")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "repeat_id": "run0",
                "trajectory_validation_status": "calibrated_visual_periodicity",
                "evaluation_artifact_status": "action_aligned_capture_ready",
                "observed_video": str(artifact),
                "observed_video_sha256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
            }
        )
    )
    assert MODULE.completed_receipt_status(
        receipt, {"repeat_id": "run1"}
    ) == ("input_mismatch_complete", False)


def test_completed_receipt_rejects_changed_execution_mode(tmp_path: Path) -> None:
    artifact = tmp_path / "observed.mp4"
    artifact.write_bytes(b"capture")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "trajectory_validation_status": "closed_loop_visual_phase",
                "evaluation_artifact_status": "action_aligned_capture_ready",
                "observed_video": str(artifact),
                "observed_video_sha256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
            }
        )
    )
    assert MODULE.completed_receipt_status(
        receipt,
        {"trajectory_validation_status": "calibrated_visual_periodicity"},
    ) == ("input_mismatch_complete", False)


def test_archive_stale_task_moves_run_recoverably(tmp_path: Path) -> None:
    source = tmp_path / "table4/RacingCar/first/run0"
    source.mkdir(parents=True)
    (source / "receipt.json").write_text('{"status":"complete"}')
    task = {
        "case_id": "table4/RacingCar/first",
        "repeat_id": "run0",
        "output_dir": str(source),
        "first_frame": "/new/frame.png",
        "actions": "/new/actions.jsonl",
        "environment_prompt": "new prompt",
        "character_prompt": "",
        "view": "first_person",
    }

    archive = MODULE.archive_stale_task(
        task, tmp_path, reason="input_mismatch_complete"
    )

    assert not source.exists()
    assert (archive / "receipt.json").is_file()
    record = json.loads((archive / "stale_reason.json").read_text())
    assert record["source"] == str(source.resolve())
    assert record["reason"] == "input_mismatch_complete"


def test_archive_stale_task_refuses_non_input_failure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="refusing to auto-archive"):
        MODULE.archive_stale_task(
            {"output_dir": str(tmp_path / "run0")},
            tmp_path,
            reason="invalid_evaluation_artifact",
        )


def test_completed_receipt_rejects_live_world_prompt_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "observed.mp4"
    artifact.write_bytes(b"capture")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "environment_prompt": "expected world",
                "view": "first_person",
                "page_url": (
                    "https://labs.google/fx/projectgenie/tools/projectgenie/w"
                    "?w=preceding+world&use3p=false"
                ),
                "trajectory_validation_status": "calibrated_visual_periodicity",
                "evaluation_artifact_status": "action_aligned_capture_ready",
                "observed_video": str(artifact),
                "observed_video_sha256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
            }
        )
    )
    assert MODULE.completed_receipt_status(
        receipt,
        {"environment_prompt": "expected world", "view": "first_person"},
    ) == ("live_world_mismatch_complete", False)


def test_completed_receipt_rejects_revisited_world(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "generation_mode": "revisit_existing_world",
                "trajectory_validation_status": "closed_loop_visual_phase",
                "evaluation_artifact_status": "native_action_aligned_ready",
            }
        )
    )
    assert MODULE.completed_receipt_status(receipt) == (
        "calibration_only_reused_world",
        False,
    )


def test_run_task_forwards_explicit_policy_auto_accept(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_run(command: list[str], check: bool):
        seen.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    task = {
        "first_frame": "/tmp/frame.png",
        "actions": "/tmp/actions.jsonl",
        "environment_prompt": "garden",
        "prompt_provenance_status": "reconstructed",
        "view": "first_person",
        "repeat_id": "run0",
        "output_dir": "/tmp/run0",
        "character_prompt": "",
        "closed_loop_yaw": False,
        "adapter_claim_scope": "reconstructed_nominal_timing",
    }
    assert MODULE.run_task(
        task,
        port=9222,
        resume=True,
        allow_canvas_entry=True,
        auto_accept_upload_policy=True,
        allow_cross_view_calibration=True,
        allow_opposite_turn_calibration=True,
    ) == 0
    assert "--auto-accept-upload-policy" in seen[0]
    scope_index = seen[0].index("--trajectory-claim-scope")
    assert seen[0][scope_index + 1] == "reconstructed_nominal_timing"
    assert "--allow-cross-view-calibration" in seen[0]
    assert "--allow-opposite-turn-calibration" in seen[0]
