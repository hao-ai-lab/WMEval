#!/usr/bin/env python3
"""Build one cross-comparison table for the calibrated active Table 4 cells."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE = {
    "table4/RacingCar/third": {
        "results": REPO_ROOT / "reports/table4/project_genie/RacingCar/third/five_run_event_aligned_error_rate_v2/per_case_results.jsonl",
        "calibration": REPO_ROOT / "runs/tables_4_7/calibration/racing_third_final.json",
        "interpretation_status": "reconstructed_pose_assumption_sensitivity",
    },
    "table4/BilliardGame/first": {
        "results": REPO_ROOT / "reports/table4/project_genie/BilliardGame/first/five_run_event_aligned_error_rate_v2/per_case_results.jsonl",
        "calibration": REPO_ROOT / "runs/tables_4_7/calibration/billiard_first_final.json",
        "interpretation_status": "diagnostic_camera_pose_leakage_risk",
    },
}
LEGACY_MECHANICS = {
    "table4/RacingCar/third": REPO_ROOT
    / "reports/table4/project_genie/RacingCar/third/five_run_reconstructed_unity_pose_common_browser_final/per_case_results.jsonl",
    "table4/BilliardGame/first": REPO_ROOT
    / "reports/table4/project_genie/BilliardGame/first/five_run_reconstructed_unity_pose_common_browser_final/per_case_results.jsonl",
}
EXCLUDED = {
    "table4/RacingCar/first": "excluded_by_user_camera_fixed_ego_motion_unavailable",
}
METRICS = {
    "dtw": "dtw_m_mean",
    "fde": "fde_m_mean",
    "error_rate_pct": "error_rate_pct",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if {str(row.get("seed")) for row in rows} != {f"run{i}" for i in range(5)}:
        raise ValueError(f"{path}: expected exactly run0 through run4")
    if any(row.get("status") != "completed" for row in rows):
        raise ValueError(f"{path}: every row must be completed")
    return sorted(rows, key=lambda row: str(row["seed"]))


def build() -> dict[str, Any]:
    baseline_path = REPO_ROOT / "experiments/protocol/paper_baseline.yaml"
    baseline = yaml.safe_load(baseline_path.read_text())
    published = baseline["tables"][4]["cases"]
    cases: dict[str, Any] = {}
    for case_id, paths in ACTIVE.items():
        rows = load_rows(paths["results"])
        calibration_path = paths.get("calibration")
        calibration = (
            json.loads(calibration_path.read_text()) if calibration_path else None
        )
        if calibration is not None and calibration.get("status") != "passed":
            raise ValueError(f"{case_id}: variance calibration has not passed")
        metrics: dict[str, Any] = {}
        fingerprints = {
            (row.get("evaluator_lock") or {}).get("evaluator_fingerprint") for row in rows
        }
        if len(fingerprints) != 1 or None in fingerprints:
            raise ValueError(f"{case_id}: rows do not share one evaluator fingerprint")
        for paper_name, rerun_name in METRICS.items():
            values = [row.get("metrics", {}).get(rerun_name) for row in rows]
            if any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in values
            ):
                raise ValueError(f"{case_id}: non-finite {rerun_name}")
            numeric = [float(value) for value in values]
            paper_mean, paper_std = published[case_id][paper_name]
            metrics[paper_name] = {
                "rerun_metric": rerun_name,
                "values": dict(zip((f"run{i}" for i in range(5)), numeric)),
                "mean": statistics.fmean(numeric),
                "sample_std": statistics.stdev(numeric),
                "published_mean": float(paper_mean),
                "published_std": float(paper_std),
                "delta_from_published_mean": statistics.fmean(numeric) - float(paper_mean),
            }
        cases[case_id] = {
            "status": "completed_reconstructed_protocol",
            "results": str(paths["results"]),
            "evaluator_fingerprint": next(iter(fingerprints)),
            "metrics": metrics,
            "variance_gate": calibration["gates"] if calibration else {},
            "interpretation_status": paths["interpretation_status"],
        }
        legacy_path = LEGACY_MECHANICS.get(case_id)
        if legacy_path is None:
            cases[case_id]["mechanics_iteration"] = {
                "current_estimator": "bounded_aligned_peak_finite_difference",
                "current_values": metrics["error_rate_pct"]["values"],
                "current_mean_error_rate_pct": metrics["error_rate_pct"]["mean"],
                "legacy_comparison_status": "not_available_for_newly_added_cell",
            }
            continue
        legacy_rows = load_rows(legacy_path)
        legacy_values = [
            float(row["metrics"]["error_rate_pct"]) for row in legacy_rows
        ]
        current_values = metrics["error_rate_pct"]["values"]
        current_mean = metrics["error_rate_pct"]["mean"]
        legacy_mean = statistics.fmean(legacy_values)
        cases[case_id]["mechanics_iteration"] = {
            "legacy_estimator": "shared_six_frame_linear_fit",
            "current_estimator": "bounded_aligned_peak_finite_difference",
            "legacy_values": dict(
                zip((f"run{i}" for i in range(5)), legacy_values)
            ),
            "current_values": current_values,
            "legacy_mean_error_rate_pct": legacy_mean,
            "current_mean_error_rate_pct": current_mean,
            "absolute_reduction_pct_points": legacy_mean - current_mean,
            "relative_reduction_fraction": (
                (legacy_mean - current_mean) / legacy_mean if legacy_mean else None
            ),
            "change_reason": (
                "The legacy shared window mixed moving and stopped states when "
                "candidate and Unity event boundaries differed by a few frames."
            ),
        }
    for case_id, reason in EXCLUDED.items():
        cases[case_id] = {"status": "excluded", "reason": reason}
    return {
        "schema_version": 1,
        "model": "Project Genie",
        "claim_scope": "paper_protocol_reconstruction_not_exact_paper_parity",
        "aggregation": "five independent generations; arithmetic mean and sample standard deviation (ddof=1)",
        "metric_note": "DTW/FDE are reconstructed Unity-plane metres; published scale transform S is unavailable, so deltas are contextual rather than exact-parity claims. Error Rate is the sole reported mechanics metric; Exact Match is intentionally omitted. Candidate pixels are lifted with timestamp-matched Unity camera poses. This is a material leakage risk when the generated camera diverges: Billiard first-person retained visual plane calibration for only 1/300 frames.",
        "cases": cases,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Calibrated Table 4 rerun",
        "",
        f"Claim scope: `{report['claim_scope']}`.",
        "",
        report["metric_note"],
        "",
        "| Case | Metric | Published mean ± std (paper units) | Rerun mean ± std | Delta | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for case_id, case in report["cases"].items():
        if case["status"] == "excluded":
            lines.append(f"| `{case_id}` | — | — | — | — | `{case['reason']}` |")
            continue
        for metric, values in case["metrics"].items():
            rerun_unit = "%" if metric.endswith("_pct") else " m"
            published_unit = "%" if metric.endswith("_pct") else ""
            lines.append(
                f"| `{case_id}` | `{metric}` | "
                f"{values['published_mean']:.4f} ± {values['published_std']:.4f}{published_unit} | "
                f"{values['mean']:.4f} ± {values['sample_std']:.4f}{rerun_unit} | "
                f"{values['delta_from_published_mean']:+.4f} | {case['interpretation_status']} |"
            )
    lines.extend(["", "## Variance gates", "", "| Case | Metric | Baseline std | Final std | Reduction | Passed |", "| --- | --- | ---: | ---: | ---: | --- |"])
    for case_id, case in report["cases"].items():
        for metric, gate in case.get("variance_gate", {}).items():
            lines.append(
                f"| `{case_id}` | `{metric}` | {gate['baseline_sample_std']:.6f} | "
                f"{gate['current_sample_std']:.6f} | {100 * gate['reduction_fraction']:.1f}% | "
                f"{'yes' if gate['passed'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Mechanics estimator iteration",
            "",
            "Exact Match is omitted. The before/after comparison changes only mechanics post-processing; object detections and tracks are identical.",
            "",
            "| Case | Legacy Error Rate | Event-aligned Error Rate | Reduction |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for case_id, case in report["cases"].items():
        iteration = case.get("mechanics_iteration")
        if not iteration or "legacy_mean_error_rate_pct" not in iteration:
            continue
        lines.append(
            f"| `{case_id}` | {iteration['legacy_mean_error_rate_pct']:.4f}% | "
            f"{iteration['current_mean_error_rate_pct']:.4f}% | "
            f"{100 * iteration['relative_reduction_fraction']:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/tables_4_7/project_genie/table4_calibrated_final",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = build()
    json_path = output / "rerun_results.json"
    markdown_path = output / "comparison.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, markdown_path)
    print(json.dumps({"status": "complete", "json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
