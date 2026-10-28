#!/usr/bin/env python3
"""Prepare and evaluate every completed, runnable Table 4/7 Project Genie cell.

The driver is deliberately resumable. Incomplete generation groups are recorded
as waiting, paper-parity blockers are recorded as blocked, and completed groups
are hash-validated, standardized, camera-pose-remapped, and passed to WMEval
Lite. It never invents missing Billiard views or unsupported browser controls.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_project_genie_case_eval import prepare as prepare_lite_case
from tools.prepare_project_genie_table7_eval import prepare_evaluation as prepare_garden
from tools.summarize_project_genie_runs import build_summary, write_summary
from tools.build_table7_rerun_results import build_results as build_table7_results
from evaluation.lite.case_loader import load_case
from evaluation.table_comparison import (
    compare_results,
    load_baseline,
    result_template,
    write_comparison,
)


@dataclass(frozen=True)
class CasePlan:
    case_id: str
    lite_case_id: str
    dimension: str
    preparation: str
    trajectory_mode: str | None = None
    reference_start_frame: int = 0
    target_yaw_deg: float = 360.0
    conditioning_frame: str | None = None


PLANS = (
    CasePlan(
        "table4/RacingCar/third",
        "RacingCar/prompt_third",
        "kinematics_mechanics",
        "racing_unity_capture",
    ),
    CasePlan(
        "table4/BilliardGame/first",
        "BilliardGame/prompt_first",
        "kinematics_mechanics",
        "billiard_kinematics_capture",
    ),
    CasePlan(
        "table7/WanderingGarden/perturb_in",
        "WanderingGarden/scene_first",
        "environment_object_consistency",
        "garden",
        "perturbation",
        0,
        -45.0,
        "experiments/tables_4_7/conditioning_frames/garden_inward.png",
    ),
    CasePlan(
        "table7/WanderingGarden/perturb_out",
        "WanderingGarden/scene_first",
        "environment_object_consistency",
        "garden",
        "perturbation",
        75,
        45.0,
        "experiments/tables_4_7/conditioning_frames/garden_outward.png",
    ),
    CasePlan(
        "table7/WanderingGarden/360_in",
        "WanderingGarden/scene_first",
        "environment_object_consistency",
        "garden",
        "full_turn",
        0,
        -360.0,
        "experiments/tables_4_7/conditioning_frames/garden_inward.png",
    ),
    CasePlan(
        "table7/WanderingGarden/360_out",
        "WanderingGarden/scene_first",
        "environment_object_consistency",
        "garden",
        "full_turn",
        75,
        360.0,
        "experiments/tables_4_7/conditioning_frames/garden_outward.png",
    ),
    CasePlan(
        "table7/BilliardGame/perturb_in",
        "BilliardGame/prompt_first",
        "environment_object_consistency",
        "billiard_unity_capture",
    ),
    CasePlan(
        "table7/BilliardGame/perturb_out",
        "BilliardGame/prompt_first",
        "environment_object_consistency",
        "billiard_unity_capture",
    ),
    CasePlan(
        "table7/BilliardGame/360_in",
        "BilliardGame/prompt_first",
        "environment_object_consistency",
        "billiard_unity_capture",
    ),
    CasePlan(
        "table7/BilliardGame/360_out",
        "BilliardGame/prompt_first",
        "environment_object_consistency",
        "billiard_unity_capture",
    ),
    CasePlan(
        "table7/MirrorRollingBall/perturb_in",
        "MirrorRollingBall/prompt_third",
        "environment_object_consistency",
        "lite",
        "perturbation",
        0,
        -45.0,
    ),
    CasePlan(
        "table7/MirrorRollingBall/perturb_out",
        "MirrorRollingBall/prompt_third",
        "environment_object_consistency",
        "lite",
        "perturbation",
        135,
        45.0,
    ),
    CasePlan(
        "table7/MirrorRollingBall/360_in",
        "MirrorRollingBall/prompt_third",
        "environment_object_consistency",
        "lite",
        "full_turn",
        0,
        -360.0,
    ),
    CasePlan(
        "table7/MirrorRollingBall/360_out",
        "MirrorRollingBall/prompt_third",
        "environment_object_consistency",
        "lite",
        "full_turn",
        135,
        360.0,
    ),
)

BLOCKED: dict[str, str] = {}

EXCLUDED: dict[str, str] = {
    "table4/RacingCar/first": (
        "Excluded from the active Table 4 evaluation by user request because "
        "the camera-fixed ego car requires unavailable video-derived ego motion; "
        "completed generation artifacts are retained for auditability."
    ),
    "table4/BilliardGame/third": (
        "Excluded from the active Table 4 suite by user request; existing raw "
        "generation and evaluation artifacts are retained for auditability."
    ),
}


def _racing_unity_params(plan: CasePlan) -> dict[str, Any]:
    if plan.case_id != "table4/RacingCar/third":
        raise ValueError(f"no dynamic Racing Unity artifact mapping for {plan.case_id}")
    root = (
        REPO_ROOT
        / "runs/tables_4_7/unity_scene_runs/captures"
        / "table4__RacingCar__third/evaluation_artifacts"
    )
    paths = {
        "kin_unity_object_transforms_json": root / "object_transforms.json",
        "kin_unity_poses_npy": root / "poses.npy",
        "kin_unity_intrinsics_npy": root / "intrinsics.npy",
        "kin_unity_object_masses_json": root / "object_masses.json",
        "kin_plane_calibration_json": root / "plane_calibration.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "dynamic Racing Unity evaluation artifacts are missing: " + ", ".join(missing)
        )
    return {
        "kin_text_prompt": "car.",
        "kin_track_point": "bbox_bottom_center",
        "kin_depth_lift": False,
        "kin_mechanics_estimator": "aligned_peak_finite_difference",
        "kin_event_alignment_tolerance_frames": 5,
        "kin_event_min_neighbor_ratio": 0.5,
        **{key: str(path.resolve()) for key, path in paths.items()},
    }


def _billiard_kinematics_inputs(plan: CasePlan) -> tuple[dict[str, Any], Path]:
    if plan.case_id != "table4/BilliardGame/first":
        raise ValueError(f"no Billiard kinematics mapping for {plan.case_id}")
    capture = (
        REPO_ROOT
        / "runs/tables_4_7/unity_scene_runs/captures"
        / plan.case_id.replace("/", "__")
    )
    artifacts = capture / "evaluation_artifacts"
    paths = {
        "kin_unity_object_transforms_json": artifacts / "object_transforms.json",
        "kin_unity_poses_npy": artifacts / "poses.npy",
        "kin_unity_intrinsics_npy": artifacts / "intrinsics.npy",
        "kin_unity_object_masses_json": artifacts / "object_masses.json",
        "kin_plane_calibration_json": artifacts / "plane_calibration.json",
    }
    reference = capture / "unity_groundtruth.mp4"
    missing = [str(path) for path in (*paths.values(), reference) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Billiard evaluation artifacts are missing: " + ", ".join(missing)
        )
    if plan.case_id.endswith("/third"):
        scene_prompt = (
            "An elevated third-person view of a carom-billiard table with a white "
            "controllable cue ball, a yellow ball, and a red ball. WASD directly "
            "controls the white ball; keep the camera above and behind the table "
            "and preserve collisions with the other balls and rails."
        )
    else:
        scene_prompt = (
            "A first-person ball-level view across a carom-billiard table with a "
            "white controllable cue ball, a yellow ball, and a red ball. WASD "
            "directly controls the white ball; the camera follows that ball while "
            "collisions with the other balls and rails are preserved."
        )
    params: dict[str, Any] = {
        "text_prompt": scene_prompt,
        "kin_text_prompt": "yellow ball.",
        "kin_track_thresh": 0.3,
        "kin_color_filter": True,
        "kin_track_point": "bbox_bottom_center",
        # Planar ray/plane intersection is the metric position source; do not
        # load a monocular depth model for this planar scene.
        "kin_depth_lift": False,
        "kin_mechanics_estimator": "aligned_peak_finite_difference",
        "kin_event_alignment_tolerance_frames": 5,
        "kin_event_min_neighbor_ratio": 0.5,
        **{key: str(path.resolve()) for key, path in paths.items()},
    }
    return params, reference


def _billiard_unity_inputs(plan: CasePlan) -> tuple[dict[str, Any], Path]:
    if plan.preparation != "billiard_unity_capture":
        raise ValueError(f"not a Billiard Unity capture plan: {plan.case_id}")
    capture = (
        REPO_ROOT
        / "runs/tables_4_7/unity_scene_runs/captures"
        / plan.case_id.replace("/", "__")
    )
    artifacts = capture / "evaluation_artifacts"
    paths = {
        "eoc_unity_object_transforms_json": artifacts / "object_transforms.json",
        "eoc_unity_poses_npy": artifacts / "poses.npy",
        "eoc_unity_intrinsics_npy": artifacts / "intrinsics.npy",
    }
    reference = capture / "unity_groundtruth.mp4"
    missing = [str(path) for path in (*paths.values(), reference) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Billiard Unity evaluation artifacts are missing: " + ", ".join(missing)
        )
    params: dict[str, Any] = {
        key: str(path.resolve()) for key, path in paths.items()
    }
    # Keep the detector prompts atomic. The original two-object prompt can be
    # returned by GroundingDINO as one merged phrase ("yellow ball red ball"),
    # which cannot match either Unity label and makes depth/distance null.
    # One environment ball plus the cue-ball character supplies the required
    # two independently labelled tracks without changing the scene geometry.
    params.update(
        {
            "eoc_text_prompt": "yellow ball.|red ball.",
            "eoc_character_prompt": "white ball.",
        }
    )
    return params, reference


def _run_root(case_id: str) -> Path:
    return REPO_ROOT / "runs" / "project_genie" / case_id


def _report_root(case_id: str) -> Path:
    table, scene, variant = case_id.split("/")
    return REPO_ROOT / "reports" / table / "project_genie" / scene / variant


def resolve_device(requested: str) -> tuple[str, str | None]:
    if requested != "auto":
        if requested == "mps":
            import torch

            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS was requested but torch.backends.mps.is_available() is false"
                )
        return requested, None
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", None
        if torch.backends.mps.is_available():
            return "mps", None
    except (ImportError, AttributeError):
        pass
    return "cpu", "No CUDA/MPS runtime is available; using CPU."


def _result_errors(results_path: Path, plan: CasePlan) -> list[str]:
    if not results_path.is_file():
        return [f"evaluator did not write {results_path}"]
    selected: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("dimension") == plan.dimension
            and row.get("case_id") == plan.lite_case_id
            and row.get("model") == "project_genie"
        ):
            selected[str(row.get("seed"))] = row
    expected = {f"run{index}" for index in range(5)}
    errors = []
    if set(selected) != expected:
        errors.append(f"expected rows {sorted(expected)}, got {sorted(selected)}")
    for repeat_id, row in selected.items():
        if row.get("status") != "completed":
            errors.append(f"{repeat_id}: {row.get('status')}: {row.get('error')}")
    return errors


def _prepare(plan: CasePlan, run_root: Path, summary_path: Path, output: Path) -> dict[str, Any]:
    if plan.preparation == "garden":
        return prepare_garden(
            validation_path=summary_path,
            unity_receipt_path=REPO_ROOT / "runs/unity/garden_bridge_gt/capture_receipt.json",
            output_dir=output,
            case_id=plan.lite_case_id,
            trajectory_mode=str(plan.trajectory_mode),
            reference_start_frame=plan.reference_start_frame,
            reference_target_yaw_deg=plan.target_yaw_deg,
            conditioning_frame_override=(
                REPO_ROOT / str(plan.conditioning_frame) if plan.conditioning_frame else None
            ),
        )
    candidate_params: dict[str, Any] | None = None
    reference_video_override: Path | None = None
    if plan.preparation == "racing_unity_capture":
        candidate_params = _racing_unity_params(plan)
    elif plan.preparation == "billiard_kinematics_capture":
        candidate_params, reference_video_override = _billiard_kinematics_inputs(plan)
    elif plan.preparation == "billiard_unity_capture":
        candidate_params, reference_video_override = _billiard_unity_inputs(plan)
    return prepare_lite_case(
        run_root,
        plan.lite_case_id,
        output,
        camera_trajectory_mode=plan.trajectory_mode,
        reference_start_frame=plan.reference_start_frame,
        reference_target_yaw_deg=plan.target_yaw_deg,
        candidate_params=candidate_params,
        reference_video_override=reference_video_override,
    )


def _write_table4_diagnostic(
    results_path: Path, output_dir: Path, target_case_id: str
) -> Path:
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = ("dtw_m_mean", "fde_m_mean", "error_rate_pct")
    values: dict[str, list[float]] = {}
    for metric in metrics:
        series = [row.get("metrics", {}).get(metric) for row in rows]
        if any(value is None for value in series):
            raise ValueError(
                f"{target_case_id}: {metric} is incomplete; inspect plane_calibration diagnostics"
            )
        values[metric] = [float(value) for value in series]
    baseline = load_baseline(REPO_ROOT / "experiments/tables_4_7/paper_baseline.yaml")
    published = baseline["tables"][4]["cases"][target_case_id]
    comparisons = {
        "dtw": {
            "published_mean": published["dtw"][0],
            "published_std": published["dtw"][1],
            "rerun_metric": "dtw_m_mean",
            "status": "reconstructed_scene_scale_comparison",
        },
        "fde": {
            "published_mean": published["fde"][0],
            "published_std": published["fde"][1],
            "rerun_metric": "fde_m_mean",
            "status": "reconstructed_scene_scale_comparison",
        },
        "error_rate_pct": {
            "published_mean": published["error_rate_pct"][0],
            "published_std": published["error_rate_pct"][1],
            "rerun_metric": "error_rate_pct",
            "status": "unity_mass_metric_trajectory_reconstruction",
        },
    }
    report = {
        "schema_version": 1,
        "status": "metric_planar_calibrated_reconstructed",
        "generation_count": len(rows),
        "reason": (
            "DTW/FDE are mapped from generated-video pixels into Unity X/Z metres "
            "by timestamp-matched Unity camera ray/plane intersection. Robust "
            "homographies fitted from tracked static conditioning-frame anchors are "
            "reported as camera-divergence QA but do not redefine metric scale. This "
            "reconstructs a deterministic common reference frame, but exact parity "
            "remains unavailable because the paper does "
            "not publish S or its exact DTW normalization. Error Rate is the sole "
            "reported mechanics metric. It follows the paper's Eq. 11 and 13 using "
            "simulator-exported mass and calibrated metric trajectories, with a "
            "bounded event-aligned finite difference preferred over windows that "
            "mix moving and stopped states."
        ),
        "metrics": {
            metric: {
                "values": dict(zip((f"run{i}" for i in range(len(rows))), series)),
                "mean": statistics.fmean(series),
                "sample_std": statistics.stdev(series),
            }
            for metric, series in values.items()
        },
        "paper_cross_comparison": comparisons,
    }
    for paper_metric in ("dtw", "fde", "error_rate_pct"):
        comparison = comparisons[paper_metric]
        rerun_summary = report["metrics"][comparison["rerun_metric"]]
        comparison["delta"] = rerun_summary["mean"] - comparison["published_mean"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostic_summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Table 4 metric reconstruction",
        "",
        report["reason"],
        "",
        "| Paper metric | Published | Rerun diagnostic | Delta | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for paper_metric, comparison in comparisons.items():
        rerun_metric = comparison["rerun_metric"]
        if rerun_metric:
            summary = report["metrics"][rerun_metric]
            unit = " m" if rerun_metric in {"dtw_m_mean", "fde_m_mean"} else "%"
            rerun = f"{summary['mean']:.4f} ± {summary['sample_std']:.4f}{unit}"
            delta = f"{comparison['delta']:+.4f}"
        else:
            rerun = "—"
            delta = "—"
        lines.append(
            f"| `{paper_metric}` | {comparison['published_mean']} ± {comparison['published_std']} | "
            f"{rerun} | {delta} | {comparison['status']} |"
        )
    (output_dir / "diagnostic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return path


def _write_comparison_artifacts(
    plan: CasePlan, prepared_dir: Path, report_dir: Path
) -> dict[str, Any]:
    results_path = report_dir / "per_case_results.jsonl"
    if plan.case_id.startswith("table4/"):
        diagnostic = _write_table4_diagnostic(results_path, report_dir, plan.case_id)
        return {"diagnostic_summary": str(diagnostic)}

    baseline_path = REPO_ROOT / "experiments/tables_4_7/paper_baseline.yaml"
    rerun = build_table7_results(
        baseline_path=baseline_path,
        results_jsonl=results_path,
        evaluation_inputs_path=prepared_dir / "evaluation_inputs.json",
        source_case_id=plan.lite_case_id,
        target_case_id=plan.case_id,
        model="project_genie",
    )
    rerun_path = report_dir / "rerun_results.json"
    rerun_path.write_text(json.dumps(rerun, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison_dir = report_dir / "comparison"
    comparison = compare_results(load_baseline(baseline_path), rerun, allow_incomplete=True)
    _write_scoped_comparison(
        comparison,
        comparison_dir,
        claim_scope=str(rerun["provenance"]["claim_scope"]),
    )
    return {
        "rerun_results": str(rerun_path),
        "comparison_json": str(comparison_dir / "comparison.json"),
        "comparison_markdown": str(comparison_dir / "comparison.md"),
        "incomplete_metrics": list(
            rerun.get("provenance", {}).get("incomplete_metrics", [])
        ),
    }


def _write_scoped_comparison(
    comparison: dict[str, Any], output_dir: Path, *, claim_scope: str
) -> None:
    """Render numeric completeness without implying exact paper parity."""
    comparison["claim_scope"] = claim_scope
    if claim_scope != "exact_paper_parity":
        for row in comparison["rows"]:
            if row.get("status") == "complete":
                row["status"] = "complete_reconstructed"
    write_comparison(comparison, output_dir)
    markdown_path = output_dir / "comparison.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    marker = "Five-run sample statistics (`ddof=1`)."
    markdown = markdown.replace(
        marker,
        f"Claim scope: `{claim_scope}`.\n\n{marker}",
        1,
    )
    markdown_path.write_text(markdown, encoding="utf-8")


def _select_consolidated_sources(
    search_root: Path,
) -> tuple[list[tuple[Path, dict[str, Any]]], dict[str, list[Any]]]:
    """Choose one latest complete, current-fingerprint result per table cell."""
    by_case: dict[str, list[tuple[float, Path, dict[str, Any]]]] = {}
    ignored: dict[str, list[Any]] = {
        "stale": [],
        "incomplete": [],
        "superseded": [],
    }
    for path in sorted(
        search_root.glob("**/five_run_reconstructed_pose_corrected_*/rerun_results.json")
    ):
        document = json.loads(path.read_text(encoding="utf-8"))
        provenance = document.get("provenance") or {}
        source_case_id = provenance.get("source_case_id")
        target_case_id = provenance.get("target_case_id")
        recorded_fingerprint = (
            (provenance.get("evaluation_settings") or {})
            .get("evaluator_lock", {})
            .get("evaluator_fingerprint")
        )
        expected_fingerprint = None
        if isinstance(source_case_id, str) and "/" in source_case_id:
            case = load_case(REPO_ROOT / "lite/cases" / source_case_id)
            expected_fingerprint = case.evaluator_lock.get("evaluator_fingerprint")
        if not expected_fingerprint or recorded_fingerprint != expected_fingerprint:
            ignored["stale"].append(
                {
                    "path": str(path.resolve()),
                    "recorded_fingerprint": recorded_fingerprint,
                    "expected_fingerprint": expected_fingerprint,
                }
            )
            continue
        incomplete = list(provenance.get("incomplete_metrics") or [])
        if incomplete:
            ignored["incomplete"].append(
                {"path": str(path.resolve()), "incomplete_metrics": incomplete}
            )
            continue
        if not isinstance(target_case_id, str):
            ignored["stale"].append(
                {"path": str(path.resolve()), "reason": "missing_target_case_id"}
            )
            continue
        by_case.setdefault(target_case_id, []).append(
            (path.stat().st_mtime, path, document)
        )
    selected: list[tuple[Path, dict[str, Any]]] = []
    for target_case_id, candidates in sorted(by_case.items()):
        _, chosen_path, chosen_document = max(candidates, key=lambda item: item[0])
        selected.append((chosen_path, chosen_document))
        for _, path, _ in candidates:
            if path != chosen_path:
                ignored["superseded"].append(
                    {
                        "path": str(path.resolve()),
                        "target_case_id": target_case_id,
                        "selected": str(chosen_path.resolve()),
                    }
                )
    return selected, ignored


def refresh_consolidated_comparison() -> dict[str, str]:
    baseline_path = REPO_ROOT / "experiments/tables_4_7/paper_baseline.yaml"
    baseline = load_baseline(baseline_path)
    merged = result_template(baseline)
    sources: list[str] = []
    search_root = REPO_ROOT / "reports/table7/project_genie"
    selected_sources, ignored_sources = _select_consolidated_sources(search_root)
    for path, document in selected_sources:
        if document.get("repeat_ids") != merged["repeat_ids"]:
            raise ValueError(f"{path}: incompatible repeat IDs")
        for case_id, metrics in document.get("results", {}).items():
            if case_id not in merged["results"] or not isinstance(metrics, dict):
                continue
            for metric, run_map in metrics.items():
                if metric not in merged["results"][case_id] or not isinstance(run_map, dict):
                    continue
                for repeat_id, value in run_map.items():
                    if value is None:
                        continue
                    prior = merged["results"][case_id][metric].get(repeat_id)
                    if prior is not None and prior != value:
                        raise ValueError(
                            f"conflicting consolidated result for {case_id}/{metric}/{repeat_id}"
                        )
                    merged["results"][case_id][metric][repeat_id] = value
        sources.append(str(path.resolve()))
    merged["provenance"] = {
        "claim_scope": "paper_protocol_reconstruction_not_exact_paper_parity",
        "source_result_files": sources,
        "ignored_stale_source_files": ignored_sources["stale"],
        "ignored_incomplete_source_files": ignored_sources["incomplete"],
        "ignored_superseded_source_files": ignored_sources["superseded"],
    }
    output_dir = REPO_ROOT / "reports/tables_4_7/project_genie/consolidated"
    output_dir.mkdir(parents=True, exist_ok=True)
    rerun_path = output_dir / "rerun_results.json"
    rerun_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    comparison = compare_results(baseline, merged, allow_incomplete=True)
    _write_scoped_comparison(
        comparison,
        output_dir,
        claim_scope=str(merged["provenance"]["claim_scope"]),
    )
    return {
        "consolidated_rerun_results": str(rerun_path),
        "consolidated_comparison_json": str(output_dir / "comparison.json"),
        "consolidated_comparison_markdown": str(output_dir / "comparison.md"),
    }


def _write_automation_status(
    *, rows: list[dict[str, Any]], device: str, prepare_only: bool, status: str
) -> dict[str, Any]:
    """Checkpoint orchestration state so an interrupted long run is inspectable."""
    report = {
        "schema_version": 1,
        "status": status,
        "prepare_only": prepare_only,
        "device": device,
        "cases": rows,
    }
    status_path = REPO_ROOT / "runs/tables_4_7/prepared_assets/evaluation_automation_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(status_path)
    return report


def run(
    *,
    selected: set[str] | None,
    device: str,
    prepare_only: bool,
    fail_incomplete: bool,
    report_suffix: str | None = None,
) -> tuple[dict[str, Any], int]:
    rows: list[dict[str, Any]] = [
        {"case_id": case_id, "status": "blocked", "reason": reason}
        for case_id, reason in BLOCKED.items()
        if selected is None or case_id in selected
    ]
    rows.extend(
        {"case_id": case_id, "status": "excluded_by_user", "reason": reason}
        for case_id, reason in EXCLUDED.items()
        if selected is None or case_id in selected
    )
    exit_code = 2 if fail_incomplete and any(row["status"] == "blocked" for row in rows) else 0
    _write_automation_status(
        rows=rows, device=device, prepare_only=prepare_only, status="running"
    )
    for plan in PLANS:
        if selected is not None and plan.case_id not in selected:
            continue
        run_root = _run_root(plan.case_id)
        summary = build_summary(run_root, [f"run{index}" for index in range(5)])
        summary_prefix = run_root / "five_run_validation"
        write_summary(summary, summary_prefix)
        if summary["status"] != "validated":
            rows.append(
                {
                    "case_id": plan.case_id,
                    "status": "waiting_for_generations",
                    "errors": summary["errors"],
                }
            )
            if fail_incomplete:
                exit_code = exit_code or 2
            _write_automation_status(
                rows=rows, device=device, prepare_only=prepare_only, status="running"
            )
            continue

        is_table4 = plan.case_id.startswith("table4/")
        prepared_dir = run_root / (
            "evaluation_prepared" if is_table4 else "evaluation_prepared_pose_corrected"
        )
        preparation = _prepare(
            plan, run_root, summary_prefix.with_suffix(".json"), prepared_dir
        )
        manifest = Path(str(preparation["candidate_manifest"])).resolve()
        report_name = (
            "five_run_reconstructed"
            if is_table4
            else f"five_run_reconstructed_pose_corrected_{device}"
        )
        if report_suffix:
            report_name += f"_{report_suffix}"
        report_dir = _report_root(plan.case_id) / report_name
        command = [
            sys.executable,
            str(REPO_ROOT / "tools/run_wmeval_lite.py"),
            "run",
            "--suite",
            str(REPO_ROOT / "lite"),
            "--candidate-manifest",
            str(manifest),
            "--dimensions",
            plan.dimension,
            "--device",
            device,
            "--resume",
            "--output-dir",
            str(report_dir),
        ]
        row: dict[str, Any] = {
            "case_id": plan.case_id,
            "status": "prepared" if prepare_only else "evaluating",
            "claim_scope": "runnable_diagnostic_not_exact_paper_parity",
            "validation_summary": str(summary_prefix.with_suffix(".json")),
            "evaluation_inputs": str(prepared_dir / "evaluation_inputs.json"),
            "candidate_manifest": str(manifest),
            "report_dir": str(report_dir),
            "command": command,
        }
        rows.append(row)
        _write_automation_status(
            rows=rows, device=device, prepare_only=prepare_only, status="running"
        )
        if not prepare_only:
            environment = os.environ.copy()
            if device == "mps":
                environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
            row["return_code"] = completed.returncode
            result_errors = _result_errors(report_dir / "per_case_results.jsonl", plan)
            if completed.returncode != 0 or result_errors:
                row["status"] = "evaluation_failed"
                row["evaluation_errors"] = result_errors
                exit_code = 1
            else:
                artifacts = _write_comparison_artifacts(plan, prepared_dir, report_dir)
                row.update(artifacts)
                incomplete_metrics = artifacts.get("incomplete_metrics", [])
                if incomplete_metrics:
                    row["status"] = "results_complete_metrics_incomplete"
                    repeat_ids = sorted(
                        {
                            str(item).split(":", 1)[0]
                            for item in incomplete_metrics
                            if str(item).startswith("run")
                        }
                    )
                    row["regeneration_required"] = [
                        {
                            "repeat_id": repeat_id,
                            "command": [
                                sys.executable,
                                str(REPO_ROOT / "tools/regenerate_invalid_project_genie_run.py"),
                                "--case", plan.case_id,
                                "--repeat", repeat_id,
                                "--reason", "required_metric_null",
                            ],
                        }
                        for repeat_id in repeat_ids
                    ]
                    if fail_incomplete:
                        exit_code = exit_code or 2
                else:
                    row["status"] = "completed"
                if plan.case_id.startswith("table7/") and not incomplete_metrics:
                    row.update(refresh_consolidated_comparison())
        _write_automation_status(
            rows=rows, device=device, prepare_only=prepare_only, status="running"
        )

    report = _write_automation_status(
        rows=rows,
        device=device,
        prepare_only=prepare_only,
        status="complete" if exit_code == 0 else "incomplete",
    )
    return report, exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--fail-incomplete",
        action="store_true",
        help=(
            "return a non-zero status when selected cells are blocked, generation "
            "groups are incomplete, or any required five-run metric is missing"
        ),
    )
    parser.add_argument(
        "--report-suffix",
        help="append a filesystem-safe attempt label to evaluator report directories",
    )
    args = parser.parse_args()
    known = {plan.case_id for plan in PLANS} | set(BLOCKED) | set(EXCLUDED)
    unknown = set(args.case) - known
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
    if args.report_suffix and not args.report_suffix.replace("_", "").replace("-", "").isalnum():
        parser.error("--report-suffix may contain only letters, numbers, hyphens, and underscores")
    try:
        device, device_note = resolve_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    report, exit_code = run(
        selected=set(args.case) or None,
        device=device,
        prepare_only=args.prepare_only,
        fail_incomplete=args.fail_incomplete,
        report_suffix=args.report_suffix,
    )
    if device_note:
        report["device_note"] = device_note
        status_path = REPO_ROOT / "runs/tables_4_7/prepared_assets/evaluation_automation_status.json"
        status_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
