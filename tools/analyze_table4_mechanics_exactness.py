#!/usr/bin/env python3
"""Analyze Table 4 mechanics exactness across stages and rounding rules.

This is a diagnostic only. It does not change the active Error-Rate-only
Table 4 contract. Candidate/Unity timelines use the bounded peak-event offset
already recorded by the event-aligned mechanics evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.modules.kinematics_mechanics import (  # noqa: E402
    NEAR_ZERO_SPEED_M_S,
    _load_camera_intrinsics,
    _load_camera_poses,
    _load_object_masses,
    _load_unity_object_transforms,
    _map_pixels_to_unity_plane_by_pose,
    _unity_trajectory,
)


CASES = {
    "table4/RacingCar/third": {
        "results": REPO_ROOT
        / "reports/table4/project_genie/RacingCar/third/five_run_event_aligned_error_rate_v2/per_case_results.jsonl",
        "artifacts": REPO_ROOT
        / "runs/tables_4_7/unity_scene_runs/captures/table4__RacingCar__third/evaluation_artifacts",
        "label": "car",
    },
    "table4/BilliardGame/first": {
        "results": REPO_ROOT
        / "reports/table4/project_genie/BilliardGame/first/five_run_event_aligned_error_rate_v2/per_case_results.jsonl",
        "artifacts": REPO_ROOT
        / "runs/tables_4_7/unity_scene_runs/captures/table4__BilliardGame__first/evaluation_artifacts",
        "label": "yellow ball",
    },
}


def _round_significant(value: float, digits: int) -> float:
    if value == 0:
        return 0.0
    decimal_places = digits - 1 - int(math.floor(math.log10(abs(value))))
    return round(value, decimal_places)


ROUNDING: dict[str, tuple[str, Callable[[float], float]]] = {
    "nearest_integer": ("nearest whole number (0 decimal places)", lambda x: round(x, 0)),
    "nearest_tenth": ("nearest tenth (1 decimal place)", lambda x: round(x, 1)),
    "one_significant_digit": ("one significant digit", lambda x: _round_significant(x, 1)),
    "two_significant_digits": ("two significant digits (paper rule)", lambda x: _round_significant(x, 2)),
}


def _speed(points: dict[int, np.ndarray], frame: int, fps: float) -> float | None:
    if frame not in points or frame - 1 not in points or fps <= 0:
        return None
    return float(np.linalg.norm(points[frame] - points[frame - 1]) * fps)


def _quantities(speed: float, mass: float) -> dict[str, float]:
    return {
        "momentum_kg_m_s": mass * speed,
        "kinetic_energy_j": 0.5 * mass * speed ** 2,
    }


def _comparison_rows(
    *,
    run_id: str,
    stage: str,
    estimated_speed: float,
    ground_truth_speed: float,
    mass: float,
    candidate_frame: int,
    ground_truth_frame: int,
) -> list[dict[str, Any]]:
    estimated = _quantities(estimated_speed, mass)
    ground_truth = _quantities(ground_truth_speed, mass)
    rows: list[dict[str, Any]] = []
    for quantity in estimated:
        rows.append(
            {
                "run_id": run_id,
                "stage": stage,
                "quantity": quantity,
                "estimated": estimated[quantity],
                "ground_truth": ground_truth[quantity],
                "candidate_frame": candidate_frame,
                "ground_truth_frame": ground_truth_frame,
                "ground_truth_speed_m_s": ground_truth_speed,
                "nontrivial_motion": ground_truth_speed >= NEAR_ZERO_SPEED_M_S,
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (description, rounder) in ROUNDING.items():
        matches = [
            rounder(float(row["estimated"])) == rounder(float(row["ground_truth"]))
            for row in rows
        ]
        output[name] = {
            "description": description,
            "match_count": sum(matches),
            "comparison_count": len(matches),
            "exact_match_pct": 100.0 * sum(matches) / len(matches) if matches else None,
        }
    return output


def analyze_case(case_id: str, config: dict[str, Any]) -> dict[str, Any]:
    results_path = Path(config["results"])
    artifacts = Path(config["artifacts"])
    label = str(config["label"])
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    transforms = _load_unity_object_transforms(str(artifacts / "object_transforms.json"))
    poses = _load_camera_poses(str(artifacts / "poses.npy"))
    intrinsics = _load_camera_intrinsics(str(artifacts / "intrinsics.npy"))
    masses = _load_object_masses(str(artifacts / "object_masses.json"))
    plane = json.loads((artifacts / "plane_calibration.json").read_text())
    plane_y = float(plane["plane"]["value_m"])
    mass = float(masses[label.lower()])
    trajectory = _unity_trajectory(transforms, label)
    if not trajectory:
        raise ValueError(f"{case_id}: Unity trajectory missing for {label}")
    ground_truth_points = {frame: point[[0, 2]] for frame, point in trajectory}

    checkpoint_rows: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    offsets: dict[str, int] = {}
    for row in rows:
        run_id = str(row["seed"])
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
        stage = km["mechanical_quantities"][label]["stages"]["aligned_peak_event"]
        offset = int(stage["temporal_offset_frames"])
        offsets[run_id] = offset
        fps = float(km["fps"])

        aligned: list[tuple[int, int, float, float]] = []
        for ground_truth_frame in sorted(ground_truth_points):
            candidate_frame = ground_truth_frame + offset
            gt_speed = _speed(ground_truth_points, ground_truth_frame, fps)
            candidate_speed = _speed(candidate_points, candidate_frame, fps)
            if gt_speed is None or candidate_speed is None:
                continue
            aligned.append(
                (ground_truth_frame, candidate_frame, gt_speed, candidate_speed)
            )
            aligned_rows.extend(
                _comparison_rows(
                    run_id=run_id,
                    stage="all_aligned_frames",
                    estimated_speed=candidate_speed,
                    ground_truth_speed=gt_speed,
                    mass=mass,
                    candidate_frame=candidate_frame,
                    ground_truth_frame=ground_truth_frame,
                )
            )
        if not aligned:
            raise ValueError(f"{case_id}/{run_id}: no aligned finite differences")

        # Initial and terminal compare each rollout's own boundary state. The
        # interior checkpoints use the fixed event offset on the common grid.
        candidate_frames = sorted(candidate_points)
        ground_truth_frames = sorted(ground_truth_points)
        boundary = {
            "initial": (
                ground_truth_frames[1],
                candidate_frames[1],
                _speed(ground_truth_points, ground_truth_frames[1], fps),
                _speed(candidate_points, candidate_frames[1], fps),
            ),
            "terminal": (
                ground_truth_frames[-1],
                candidate_frames[-1],
                _speed(ground_truth_points, ground_truth_frames[-1], fps),
                _speed(candidate_points, candidate_frames[-1], fps),
            ),
        }
        for stage_name, values in boundary.items():
            gt_frame, candidate_frame, gt_speed, candidate_speed = values
            if gt_speed is None or candidate_speed is None:
                continue
            checkpoint_rows.extend(
                _comparison_rows(
                    run_id=run_id,
                    stage=stage_name,
                    estimated_speed=candidate_speed,
                    ground_truth_speed=gt_speed,
                    mass=mass,
                    candidate_frame=candidate_frame,
                    ground_truth_frame=gt_frame,
                )
            )

        for stage_name, fraction in (
            ("quarter", 0.25),
            ("midpoint", 0.50),
            ("three_quarter", 0.75),
        ):
            index = int(round((len(aligned) - 1) * fraction))
            gt_frame, candidate_frame, gt_speed, candidate_speed = aligned[index]
            checkpoint_rows.extend(
                _comparison_rows(
                    run_id=run_id,
                    stage=stage_name,
                    estimated_speed=candidate_speed,
                    ground_truth_speed=gt_speed,
                    mass=mass,
                    candidate_frame=candidate_frame,
                    ground_truth_frame=gt_frame,
                )
            )

        peak_gt_speed = float(stage["ground_truth_speed_m_s"])
        peak_candidate_speed = float(stage["candidate_speed_m_s"])
        checkpoint_rows.extend(
            _comparison_rows(
                run_id=run_id,
                stage="aligned_peak",
                estimated_speed=peak_candidate_speed,
                ground_truth_speed=peak_gt_speed,
                mass=mass,
                candidate_frame=int(stage["candidate_peak_frame"]),
                ground_truth_frame=int(stage["ground_truth_peak_frame"]),
            )
        )

    by_stage: dict[str, Any] = {}
    for stage in sorted({str(row["stage"]) for row in checkpoint_rows}):
        stage_rows = [row for row in checkpoint_rows if row["stage"] == stage]
        by_stage[stage] = {
            "all_quantities": _summarize(stage_rows),
            "momentum": _summarize(
                [row for row in stage_rows if row["quantity"] == "momentum_kg_m_s"]
            ),
            "kinetic_energy": _summarize(
                [row for row in stage_rows if row["quantity"] == "kinetic_energy_j"]
            ),
            "comparison_count": len(stage_rows),
        }
    nontrivial_aligned = [row for row in aligned_rows if row["nontrivial_motion"]]
    checkpoint_by_quantity = {
        quantity: _summarize(
            [row for row in checkpoint_rows if row["quantity"] == key]
        )
        for quantity, key in (
            ("momentum", "momentum_kg_m_s"),
            ("kinetic_energy", "kinetic_energy_j"),
        )
    }
    nontrivial_by_quantity = {
        quantity: _summarize(
            [row for row in nontrivial_aligned if row["quantity"] == key]
        )
        for quantity, key in (
            ("momentum", "momentum_kg_m_s"),
            ("kinetic_energy", "kinetic_energy_j"),
        )
    }
    checkpoint_exactness_by_run = {
        run_id: _summarize(
            [row for row in checkpoint_rows if row["run_id"] == run_id]
        )
        for run_id in sorted(offsets)
    }
    five_run_checkpoint_exactness: dict[str, Any] = {}
    for rounding_name in ROUNDING:
        values = [
            float(checkpoint_exactness_by_run[run_id][rounding_name]["exact_match_pct"])
            for run_id in sorted(checkpoint_exactness_by_run)
        ]
        five_run_checkpoint_exactness[rounding_name] = {
            "values_pct": dict(zip(sorted(checkpoint_exactness_by_run), values)),
            "mean_pct": statistics.fmean(values),
            "sample_std_pct": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return {
        "source_results": str(results_path),
        "label": label,
        "mass_kg": mass,
        "run_count": len(rows),
        "temporal_offsets_frames": offsets,
        "checkpoint_count": len(checkpoint_rows),
        "aligned_comparison_count": len(aligned_rows),
        "nontrivial_aligned_comparison_count": len(nontrivial_aligned),
        "checkpoint_exactness": _summarize(checkpoint_rows),
        "checkpoint_exactness_by_run": checkpoint_exactness_by_run,
        "five_run_checkpoint_exactness": five_run_checkpoint_exactness,
        "checkpoint_exactness_by_quantity": checkpoint_by_quantity,
        "checkpoint_exactness_by_stage": by_stage,
        "all_aligned_frames_exactness": _summarize(aligned_rows),
        "nontrivial_motion_exactness": _summarize(nontrivial_aligned),
        "nontrivial_motion_exactness_by_quantity": nontrivial_by_quantity,
        "checkpoint_rows": checkpoint_rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Table 4 mechanics exactness diagnostic",
        "",
        "This diagnostic does not change the active Error-Rate-only Table 4 contract.",
        "",
        "Whole-number and tenth-place rounding are unit-scale dependent; significant-digit rounding is scale invariant. Initial and terminal checkpoints are reported separately because stationary states can make coarse rounding trivially exact.",
        "",
        "## Aggregate results",
        "",
        "| Case | Population | Whole number | Tenth | 1 significant digit | 2 significant digits |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    columns = (
        "nearest_integer",
        "nearest_tenth",
        "one_significant_digit",
        "two_significant_digits",
    )
    for case_id, case in report["cases"].items():
        for population, key in (
            ("six checkpoints", "checkpoint_exactness"),
            ("all aligned frames", "all_aligned_frames_exactness"),
            ("nontrivial aligned motion", "nontrivial_motion_exactness"),
        ):
            result = case[key]
            cells = [
                "—"
                if result[column]["exact_match_pct"] is None
                else f"{result[column]['exact_match_pct']:.2f}%"
                for column in columns
            ]
            lines.append(
                f"| `{case_id}` | {population} | " + " | ".join(cells) + " |"
            )
    lines.extend(["", "## Checkpoint breakdown", ""])
    for case_id, case in report["cases"].items():
        lines.extend(
            [
                f"### {case_id}",
                "",
                "| Stage | Whole number | Tenth | 1 significant digit | 2 significant digits |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for stage, stage_result in case["checkpoint_exactness_by_stage"].items():
            result = stage_result["all_quantities"]
            cells = [f"{result[column]['exact_match_pct']:.2f}%" for column in columns]
            lines.append(f"| `{stage}` | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(["## Quantity breakdown", ""])
    for case_id, case in report["cases"].items():
        lines.extend(
            [
                f"### {case_id}",
                "",
                "| Population | Quantity | Whole number | Tenth | 1 significant digit | 2 significant digits |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for population, key in (
            ("six checkpoints", "checkpoint_exactness_by_quantity"),
            ("nontrivial aligned motion", "nontrivial_motion_exactness_by_quantity"),
        ):
            for quantity, result in case[key].items():
                cells = [
                    f"{result[column]['exact_match_pct']:.2f}%" for column in columns
                ]
                lines.append(
                    f"| {population} | `{quantity}` | "
                    + " | ".join(cells)
                    + " |"
                )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "reports/tables_4_7/project_genie/table4_mechanics_exactness",
    )
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "official_table4_mechanics_metric": "error_rate_pct",
        "rounding_rules": {
            name: description for name, (description, _) in ROUNDING.items()
        },
        "sampling": {
            "checkpoints": [
                "initial",
                "quarter",
                "midpoint",
                "three_quarter",
                "aligned_peak",
                "terminal",
            ],
            "aligned_frames": "candidate frame = Unity frame + fixed peak-event offset",
            "nontrivial_motion_threshold_m_s": NEAR_ZERO_SPEED_M_S,
        },
        "cases": {
            case_id: analyze_case(case_id, config)
            for case_id, config in CASES.items()
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "analysis.json"
    markdown_path = output / "analysis.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, markdown_path)
    print(
        json.dumps(
            {"status": "complete", "json": str(json_path), "markdown": str(markdown_path)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
