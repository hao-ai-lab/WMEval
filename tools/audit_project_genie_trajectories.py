#!/usr/bin/env python3
"""Audit Project Genie Table 4/7 action traces against one execution policy.

The audit is intentionally stricter than receipt completeness.  It checks the
compiled action program, the actually dispatched browser operations, capture
coverage, artifact hashes, repeat consistency, and the standardized adapter:

* Table 4 racing uses the declared reconstructed nominal schedule.
* Table 7 yaw uses the validated backend-period calibration for both 45 and
  360 degree motion, preserving the ten-second nominal horizon.

Any run that violates the policy is emitted as a selective regeneration
candidate.  Source runs are never modified by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "project_genie"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def read_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: expected a list of objects")
    return value


def read_actions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}: action must be an object")
        rows.append(row)
    return rows


def resolved_file(run_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("missing path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (run_dir / path).resolve()


def expected_mode(case_id: str) -> str:
    if case_id.startswith("table7/"):
        return "calibrated_visual_periodicity"
    return "reconstructed_nominal_timing"


def expected_actions(receipt: dict[str, Any], source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode = receipt.get("trajectory_validation_status")
    if mode == "calibrated_visual_periodicity":
        calibrated = receipt.get("calibrated_actions")
        if not isinstance(calibrated, list) or not all(
            isinstance(row, dict) for row in calibrated
        ):
            raise ValueError("calibrated receipt lacks calibrated_actions")
        return calibrated
    return source


def audit_run(case_id: str, run_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    receipt_path = run_dir / "receipt.json"
    try:
        receipt = read_object(receipt_path)
    except Exception as exc:
        return {
            "run": str(run_dir),
            "repeat_id": run_dir.name,
            "status": "invalid",
            "issues": [f"receipt: {exc}"],
        }

    mode = str(receipt.get("trajectory_validation_status", "missing"))
    wanted_mode = expected_mode(case_id)
    if receipt.get("status") != "complete":
        issues.append("receipt_not_complete")
    if receipt.get("table_repeat_eligible") is not True:
        issues.append("not_table_repeat_eligible")
    if receipt.get("generation_mode") != "new_world_generation":
        issues.append("not_independent_generation")
    if mode != wanted_mode:
        issues.append(f"execution_mode:{mode};expected:{wanted_mode}")

    trace: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    try:
        actions_path = resolved_file(run_dir, receipt.get("actions"))
        if sha256(actions_path) != receipt.get("actions_sha256"):
            issues.append("actions_hash_mismatch")
        source_actions = read_actions(actions_path)
        expected = expected_actions(receipt, source_actions)

        trace_path = resolved_file(run_dir, receipt.get("action_trace"))
        if sha256(trace_path) != receipt.get("action_trace_sha256"):
            issues.append("action_trace_hash_mismatch")
        trace = read_list(trace_path)

        frame_manifest_path = resolved_file(run_dir, receipt.get("observed_frames"))
        if sha256(frame_manifest_path) != receipt.get("observed_frames_sha256"):
            issues.append("observed_frames_hash_mismatch")
        manifest = read_object(frame_manifest_path)
        raw_frames = manifest.get("frames")
        if not isinstance(raw_frames, list) or not all(
            isinstance(row, dict) for row in raw_frames
        ):
            raise ValueError("observed frame manifest lacks frames")
        frames = raw_frames

        observed_video = resolved_file(run_dir, receipt.get("observed_video"))
        if sha256(observed_video) != receipt.get("observed_video_sha256"):
            issues.append("observed_video_hash_mismatch")
    except Exception as exc:
        issues.append(f"artifact_validation:{exc}")

    if expected and len(trace) != len(expected):
        issues.append(f"action_count:{len(trace)};expected:{len(expected)}")
    for index, (wanted, actual) in enumerate(zip(expected, trace)):
        if actual.get("index") != index:
            issues.append(f"action_{index}:index_mismatch")
        if actual.get("op") != wanted.get("op"):
            issues.append(f"action_{index}:op_mismatch")
        wanted_keys = [str(key) for key in wanted.get("keys", [])]
        actual_keys = [str(key) for key in actual.get("keys", [])]
        if actual_keys != wanted_keys:
            issues.append(f"action_{index}:keys_mismatch")
        wanted_ms = int(wanted.get("ms", 0))
        actual_ms = int(actual.get("duration_ms", -1))
        if actual_ms != wanted_ms:
            issues.append(f"action_{index}:duration:{actual_ms};expected:{wanted_ms}")
        if "target_yaw_deg" in wanted and float(actual.get("target_yaw_deg", 0.0)) != float(
            wanted["target_yaw_deg"]
        ):
            issues.append(f"action_{index}:yaw_target_mismatch")
        try:
            dispatched = float(actual["dispatched_at_sec"])
            accepted = float(actual["accepted_at_sec"])
            completed = float(actual["completed_at_sec"])
            if not dispatched <= accepted <= completed:
                issues.append(f"action_{index}:non_monotonic_timestamps")
            elapsed_ms = (completed - accepted) * 1000.0
            tolerance_ms = max(250.0, wanted_ms * 0.20)
            if abs(elapsed_ms - wanted_ms) > tolerance_ms:
                issues.append(f"action_{index}:execution_timing_drift")
        except (KeyError, TypeError, ValueError):
            issues.append(f"action_{index}:invalid_timestamps")

    if frames:
        elapsed = [float(row.get("elapsed_sec", -1.0)) for row in frames]
        if any(right <= left for left, right in zip(elapsed, elapsed[1:])):
            issues.append("capture_timeline_not_strictly_increasing")
        if len(frames) != receipt.get("observed_frame_count"):
            issues.append("observed_frame_count_mismatch")
        if abs(elapsed[-1] - float(receipt.get("observed_duration_sec", -1.0))) > 0.05:
            issues.append("observed_duration_mismatch")
        if trace and abs(elapsed[-1] - float(trace[-1].get("completed_at_sec", -1.0))) > 0.25:
            issues.append("capture_does_not_cover_action_endpoint")
        for index, action in enumerate(trace):
            start = float(action.get("accepted_at_sec", -1.0))
            end = float(action.get("completed_at_sec", -1.0))
            if not any(start <= value <= end for value in elapsed):
                issues.append(f"action_{index}:no_capture_coverage")

    calibrated_turn_ms: int | None = None
    target_yaw: float | None = None
    for row in expected:
        if "target_yaw_deg" in row:
            calibrated_turn_ms = int(row.get("ms", 0))
            target_yaw = float(row["target_yaw_deg"])
            break
    nominal_horizon_ms = sum(int(row.get("ms", 0)) for row in expected)
    if case_id.startswith("table7/") and expected:
        if nominal_horizon_ms != 10_000:
            issues.append(f"nominal_horizon_ms:{nominal_horizon_ms};expected:10000")
        if calibrated_turn_ms is None or target_yaw is None:
            issues.append("missing_calibrated_yaw_action")
        else:
            calibration = receipt.get("turn_calibration")
            period_sec = calibration.get("full_turn_duration_sec") if isinstance(calibration, dict) else None
            if not isinstance(period_sec, (int, float)):
                issues.append("missing_turn_calibration_period")
            else:
                expected_ms = int(round(float(period_sec) * 1000.0 * abs(target_yaw) / 360.0))
                if calibrated_turn_ms != expected_ms:
                    issues.append(
                        f"calibrated_turn_ms:{calibrated_turn_ms};expected:{expected_ms}"
                    )

    video_hash = receipt.get("evaluation_video_sha256") or receipt.get(
        "observed_video_sha256"
    )
    return {
        "run": str(run_dir.resolve()),
        "repeat_id": run_dir.name,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "execution_mode": mode,
        "expected_execution_mode": wanted_mode,
        "action_count": len(trace),
        "capture_frame_count": len(frames),
        "observed_duration_sec": receipt.get("observed_duration_sec"),
        "calibrated_turn_ms": calibrated_turn_ms,
        "nominal_horizon_ms": nominal_horizon_ms,
        "first_frame_sha256": receipt.get("first_frame_sha256"),
        "actions_sha256": receipt.get("actions_sha256"),
        "evaluation_video_sha256": video_hash,
    }


def discover_cells(run_root: Path) -> list[tuple[str, Path]]:
    cells: list[tuple[str, Path]] = []
    for table in ("table4", "table7"):
        table_root = run_root / table
        if not table_root.is_dir():
            continue
        for scene in sorted(path for path in table_root.iterdir() if path.is_dir()):
            for variant in sorted(path for path in scene.iterdir() if path.is_dir()):
                if any((variant / f"run{index}" / "receipt.json").is_file() for index in range(5)):
                    cells.append((f"{table}/{scene.name}/{variant.name}", variant))
    return cells


def build_audit(run_root: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    candidates: list[str] = []
    for case_id, cell_root in discover_cells(run_root):
        runs = [audit_run(case_id, cell_root / f"run{index}") for index in range(5)]
        cell_issues: list[str] = []
        if len(runs) != 5:
            cell_issues.append("repeat_count_not_five")
        first_hashes = {row.get("first_frame_sha256") for row in runs}
        action_hashes = {row.get("actions_sha256") for row in runs}
        output_hashes = {row.get("evaluation_video_sha256") for row in runs}
        if len(first_hashes) != 1:
            cell_issues.append("first_frame_hashes_differ")
        if len(action_hashes) != 1:
            cell_issues.append("action_hashes_differ")
        if None in output_hashes or len(output_hashes) != len(runs):
            cell_issues.append("evaluation_outputs_not_five_unique_hashes")
        durations = [
            float(row["observed_duration_sec"])
            for row in runs
            if isinstance(row.get("observed_duration_sec"), (int, float))
        ]
        invalid_runs = [row for row in runs if row["status"] != "valid"]
        candidates.extend(row["run"] for row in invalid_runs)
        cells.append(
            {
                "case_id": case_id,
                "status": "valid" if not cell_issues and not invalid_runs else "invalid",
                "issues": cell_issues,
                "duration_mean_sec": statistics.mean(durations) if durations else None,
                "duration_sample_std_sec": (
                    statistics.stdev(durations) if len(durations) > 1 else 0.0
                ),
                "runs": runs,
            }
        )
    invalid_cells = [row for row in cells if row["status"] != "valid"]
    return {
        "schema_version": 1,
        "policy": "table4_nominal_table7_calibrated_fixed_10s_v1",
        "status": "valid" if not invalid_cells else "regeneration_required",
        "run_root": str(run_root.resolve()),
        "cell_count": len(cells),
        "run_count": sum(len(row["runs"]) for row in cells),
        "valid_cell_count": len(cells) - len(invalid_cells),
        "invalid_cell_count": len(invalid_cells),
        "regeneration_candidate_count": len(candidates),
        "regeneration_candidates": candidates,
        "cells": cells,
    }


def write_outputs(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = output.with_suffix(".md")
    lines = [
        "# Project Genie trajectory audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Case | Status | Mean duration (s) | Invalid runs |",
        "| --- | --- | ---: | ---: |",
    ]
    for cell in report["cells"]:
        invalid = sum(run["status"] != "valid" for run in cell["runs"])
        mean = cell["duration_mean_sec"]
        lines.append(
            f"| {cell['case_id']} | {cell['status']} | "
            f"{mean:.3f} | {invalid} |"
        )
    lines.extend(
        [
            "",
            f"Selective regeneration candidates: **{report['regeneration_candidate_count']}**.",
            "",
            "The audit validates execution and provenance, not exact unpublished paper timing.",
            "",
        ]
    )
    markdown.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "runs" / "tables_4_7" / "project_genie_trajectory_audit.json",
    )
    args = parser.parse_args()
    report = build_audit(args.run_root.resolve())
    write_outputs(report, args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
