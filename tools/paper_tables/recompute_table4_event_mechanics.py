#!/usr/bin/env python3
"""Recompute Table 4 mechanics from cached tracks with event alignment.

Object detection and tracking are unchanged, so this tool avoids paying for a
second GroundingDINO pass when only the mechanics estimator changes.  It writes
new reports, records source hashes, and never mutates the source evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.modules.kinematics_mechanics import (  # noqa: E402
    _aligned_peak_event_stage,
    _load_camera_intrinsics,
    _load_camera_poses,
    _load_object_masses,
    _load_unity_object_transforms,
    _map_pixels_to_unity_plane_by_pose,
    _summarize_mechanical_quantities,
    _unity_trajectory,
)
from tools.paper_tables.run_prepared_table_evaluations import _write_table4_diagnostic  # noqa: E402


CASES = {
    "table4/RacingCar/third": {
        "source": REPO_ROOT
        / "reports/table4/project_genie/RacingCar/third/five_run_reconstructed_unity_pose_common_browser_final/per_case_results.jsonl",
        "output": REPO_ROOT
        / "reports/table4/project_genie/RacingCar/third/five_run_event_aligned_error_rate_v2",
        "artifacts": REPO_ROOT
        / "runs/tables_4_7/unity_scene_runs/captures/table4__RacingCar__third/evaluation_artifacts",
        "label": "car",
    },
    "table4/BilliardGame/first": {
        "source": REPO_ROOT
        / "reports/table4/project_genie/BilliardGame/first/five_run_reconstructed_unity_pose_common_browser_final/per_case_results.jsonl",
        "output": REPO_ROOT
        / "reports/table4/project_genie/BilliardGame/first/five_run_event_aligned_error_rate_v2",
        "artifacts": REPO_ROOT
        / "runs/tables_4_7/unity_scene_runs/captures/table4__BilliardGame__first/evaluation_artifacts",
        "label": "yellow ball",
    },
}

METHOD = {
    "name": "aligned_peak_finite_difference",
    "event_alignment_tolerance_frames": 5,
    "persistence_min_neighbor_ratio": 0.5,
    "reporting_policy": "error_rate_only",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(previous: str, case_id: str) -> str:
    payload = {
        "case_id": case_id,
        "previous_fingerprint": previous,
        "method": METHOD,
        "module_sha256": _sha256(
            REPO_ROOT / "evaluation/modules/kinematics_mechanics.py"
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:32]


def _remove_exact_match(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("exact_match_pct", None)
        for child in value.values():
            _remove_exact_match(child)
    elif isinstance(value, list):
        for child in value:
            _remove_exact_match(child)


def recompute_case(case_id: str, config: dict[str, Any]) -> Path:
    source = Path(config["source"])
    output = Path(config["output"])
    artifacts = Path(config["artifacts"])
    label = str(config["label"])
    if not source.is_file():
        raise FileNotFoundError(source)

    transforms = _load_unity_object_transforms(str(artifacts / "object_transforms.json"))
    poses = _load_camera_poses(str(artifacts / "poses.npy"))
    intrinsics = _load_camera_intrinsics(str(artifacts / "intrinsics.npy"))
    masses = _load_object_masses(str(artifacts / "object_masses.json"))
    plane = json.loads((artifacts / "plane_calibration.json").read_text())
    plane_y = float(plane["plane"]["value_m"])
    trajectory = _unity_trajectory(transforms, label)
    if not trajectory:
        raise ValueError(f"{case_id}: missing Unity trajectory for {label}")
    ground_truth_points = {frame: point[[0, 2]] for frame, point in trajectory}
    mass_kg = float(masses[label.lower()])

    source_lines = [line for line in source.read_text().splitlines() if line.strip()]
    if len(source_lines) != 5:
        raise ValueError(f"{source}: expected five source rows")
    rows: list[dict[str, Any]] = []
    for source_line in source_lines:
        row = json.loads(source_line)
        km = row["full_report"]["results"]["kinematics_mechanics"]
        tracked = km["trajectories"][label]
        point_key = (
            "bbox_bottom_center_uv"
            if km.get("track_point") == "bbox_bottom_center"
            else "centroid_uv"
        )
        pixels = {
            int(point["frame"]): np.asarray(point[point_key], dtype=np.float64)
            for point in tracked
        }
        candidate_points = _map_pixels_to_unity_plane_by_pose(
            pixels, poses, intrinsics, plane_y
        )
        stage = _aligned_peak_event_stage(
            label,
            mass_kg,
            candidate_points,
            ground_truth_points,
            float(km["fps"]),
            tolerance_frames=METHOD["event_alignment_tolerance_frames"],
            min_neighbor_ratio=METHOD["persistence_min_neighbor_ratio"],
        )
        if stage is None:
            raise ValueError(f"{case_id}/{row['seed']}: no valid aligned peak event")
        mechanics = km["mechanical_quantities"]
        mechanics[label]["stages"]["aligned_peak_event"] = stage
        mechanics[label]["stage_selection_for_summary"] = (
            "aligned_peak_event_then_peak_motion_then_terminal"
        )
        summary = _summarize_mechanical_quantities(mechanics)
        if summary is None:
            raise ValueError(f"{case_id}/{row['seed']}: mechanical summary is empty")
        summary["reporting_policy"] = METHOD["reporting_policy"]
        summary["method_note"] = (
            "Peak candidate and Unity events are selected independently by sustained "
            f"motion within a fixed +/-{METHOD['event_alignment_tolerance_frames']}-frame "
            "alignment tolerance. No ground-truth "
            "value is used to choose the candidate speed."
        )
        km["mechanical_summary"] = summary
        km["mechanics_estimator"] = METHOD["name"]
        km["event_alignment_tolerance_frames"] = METHOD[
            "event_alignment_tolerance_frames"
        ]
        row.setdefault("metrics", {})["error_rate_pct"] = summary["error_rate_pct"]
        row["metrics"].pop("exact_match_pct", None)
        old_fingerprint = str(row["evaluator_lock"]["evaluator_fingerprint"])
        row["evaluator_lock"]["evaluator_fingerprint"] = _fingerprint(
            old_fingerprint, case_id
        )
        row["evaluator_lock"]["release_id"] = (
            "wmeval-lite-v1-table4-event-aligned-error-rate-v2"
        )
        row["evaluator_lock"]["module_config"] = dict(METHOD)
        row["mechanics_reprocessing"] = {
            "source_results": str(source),
            "source_row_sha256": hashlib.sha256(source_line.encode()).hexdigest(),
            "source_results_sha256": _sha256(source),
            "source_evaluator_fingerprint": old_fingerprint,
            "method": dict(METHOD),
        }
        _remove_exact_match(km["mechanical_quantities"])
        _remove_exact_match(km["mechanical_summary"])
        rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    destination = output / "per_case_results.jsonl"
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_table4_diagnostic(destination, output, case_id)
    return destination


def write_sensitivity(path: Path) -> Path:
    tolerances = [0, 1, 2, 3, 5, 10, 15, 30]
    cases: dict[str, Any] = {}
    for case_id, config in CASES.items():
        source = Path(config["source"])
        artifacts = Path(config["artifacts"])
        label = str(config["label"])
        transforms = _load_unity_object_transforms(
            str(artifacts / "object_transforms.json")
        )
        poses = _load_camera_poses(str(artifacts / "poses.npy"))
        intrinsics = _load_camera_intrinsics(str(artifacts / "intrinsics.npy"))
        masses = _load_object_masses(str(artifacts / "object_masses.json"))
        plane = json.loads((artifacts / "plane_calibration.json").read_text())
        plane_y = float(plane["plane"]["value_m"])
        trajectory = _unity_trajectory(transforms, label)
        if not trajectory:
            raise ValueError(f"{case_id}: missing Unity trajectory for {label}")
        ground_truth_points = {frame: point[[0, 2]] for frame, point in trajectory}
        mass_kg = float(masses[label.lower()])
        rows = [
            json.loads(line)
            for line in source.read_text().splitlines()
            if line.strip()
        ]
        tolerance_rows: dict[str, Any] = {}
        for tolerance in tolerances:
            values: dict[str, float | None] = {}
            offsets: dict[str, int | None] = {}
            for row in rows:
                km = row["full_report"]["results"]["kinematics_mechanics"]
                tracked = km["trajectories"][label]
                point_key = (
                    "bbox_bottom_center_uv"
                    if km.get("track_point") == "bbox_bottom_center"
                    else "centroid_uv"
                )
                pixels = {
                    int(point["frame"]): np.asarray(
                        point[point_key], dtype=np.float64
                    )
                    for point in tracked
                }
                candidate_points = _map_pixels_to_unity_plane_by_pose(
                    pixels, poses, intrinsics, plane_y
                )
                stage = _aligned_peak_event_stage(
                    label,
                    mass_kg,
                    candidate_points,
                    ground_truth_points,
                    float(km["fps"]),
                    tolerance_frames=tolerance,
                    min_neighbor_ratio=METHOD["persistence_min_neighbor_ratio"],
                )
                run_id = str(row["seed"])
                if stage is None:
                    values[run_id] = None
                    offsets[run_id] = None
                    continue
                quantity_errors = [
                    float(stage[name]["error_rate_pct"])
                    for name in ("momentum_kg_m_s", "kinetic_energy_j")
                ]
                values[run_id] = statistics.fmean(quantity_errors)
                offsets[run_id] = int(stage["temporal_offset_frames"])
            numeric = [value for value in values.values() if value is not None]
            tolerance_rows[str(tolerance)] = {
                "complete_run_count": len(numeric),
                "mean_error_rate_pct": (
                    statistics.fmean(numeric) if len(numeric) == len(rows) else None
                ),
                "sample_std_error_rate_pct": (
                    statistics.stdev(numeric) if len(numeric) == len(rows) else None
                ),
                "values": values,
                "temporal_offsets_frames": offsets,
            }
        cases[case_id] = {"tolerances": tolerance_rows}
    report = {
        "schema_version": 1,
        "method": METHOD,
        "interpretation": (
            "Tolerance is fixed before scoring and only bounds event timing. "
            "The candidate peak is selected by sustained motion, not by matching "
            "the Unity speed."
        ),
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    args = parser.parse_args()
    selected = args.case or sorted(CASES)
    outputs = {
        case_id: str(recompute_case(case_id, CASES[case_id]))
        for case_id in selected
    }
    sensitivity = write_sensitivity(
        REPO_ROOT
        / "runs/tables_4_7/calibration/table4_event_alignment_sensitivity.json"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "outputs": outputs,
                "sensitivity": str(sensitivity),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
