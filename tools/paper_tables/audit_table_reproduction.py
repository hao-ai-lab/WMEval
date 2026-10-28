#!/usr/bin/env python3
"""Audit end-to-end Table 4/7 reconstruction and execution readiness.

This is intentionally stricter than checking that files merely exist.  It
connects the paper-source hashes, semantic trajectory contract, generated
browser receipts, current evaluator fingerprints, and five-repeat result rows
into one resumable status artifact.  Exact paper parity remains a separate
claim and is never inferred from a successful reconstructed-protocol run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.lite.case_loader import load_case
from tools.trajectories.audit_trajectory_readiness import inventory
from tools.trajectories.prepare_table_trajectories import audit as audit_table_trajectories
from tools.paper_tables.run_prepared_table_evaluations import BLOCKED, EXCLUDED, PLANS
from tools.paper_tables.run_project_genie_tables import completed_receipt_status

TABLE_ROOT = REPO_ROOT / "experiments" / "protocol"
ASSET_ROOT = REPO_ROOT / "runs" / "tables_4_7" / "prepared_assets"
EXPECTED_REPEATS = {f"run{index}" for index in range(5)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _paper_sources() -> dict[str, Any]:
    baseline_path = TABLE_ROOT / "paper_baseline.yaml"
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    for source_id, source in baseline.get("sources", {}).items():
        path = (baseline_path.parent / str(source["path"])).resolve()
        expected = str(source["sha256"])
        actual = _sha256(path) if path.is_file() else None
        rows[str(source_id)] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "verified" if actual == expected else "mismatch_or_missing",
        }
    return {
        "status": (
            "verified"
            if rows and all(row["status"] == "verified" for row in rows.values())
            else "failed"
        ),
        "sources": rows,
    }


def _generation_status() -> dict[str, Any]:
    manifest = _read_json(ASSET_ROOT / "asset_manifest.json")
    rows: list[dict[str, Any]] = []
    for case in manifest.get("cases", []):
        case_id = str(case["case_id"])
        generation_status = str(case["status"])
        if generation_status == "excluded":
            rows.append(
                {
                    "case_id": case_id,
                    "status": "excluded_by_execution_scope",
                    "issues": case.get("issues", []),
                }
            )
            continue
        if generation_status == "blocked":
            rows.append(
                {
                    "case_id": case_id,
                    "status": "blocked_by_declared_inputs_or_controls",
                    "issues": case.get("issues", []),
                }
            )
            continue
        validation_path = (
            REPO_ROOT / "runs" / "project_genie" / case_id / "five_run_validation.json"
        )
        if not validation_path.is_file():
            rows.append(
                {
                    "case_id": case_id,
                    "status": "missing_validation",
                    "validation": str(validation_path),
                }
            )
            continue
        validation = _read_json(validation_path)
        repeat_ids = set(str(value) for value in validation.get("repeat_ids", []))
        conditioning = case.get("conditioning") or {}
        actions = case.get("actions") or {}
        action_files = actions.get("files") or {}
        browser_action = action_files.get(
            f"{actions.get('trajectory_stem')}.actions.jsonl", {}
        )
        receipt_rows: list[dict[str, Any]] = []
        for repeat_id in sorted(EXPECTED_REPEATS):
            receipt_path = (
                REPO_ROOT / "runs" / "project_genie" / case_id / repeat_id / "receipt.json"
            )
            if receipt_path.is_file():
                receipt_status, receipt_valid = completed_receipt_status(
                    receipt_path,
                    {
                        "repeat_id": repeat_id,
                        "view": case.get("view"),
                        "environment_prompt": case.get("environment_prompt"),
                        "character_prompt": case.get("character_prompt"),
                        "prompt_provenance_status": case.get(
                            "prompt_provenance_status"
                        ),
                        "actions_sha256": browser_action.get("sha256"),
                        "first_frame_sha256": conditioning.get("sha256"),
                    },
                )
            else:
                receipt_status, receipt_valid = "missing_receipt", False
            receipt_rows.append(
                {
                    "repeat_id": repeat_id,
                    "receipt": str(receipt_path),
                    "status": receipt_status,
                    "valid": receipt_valid,
                }
            )
        valid = (
            validation.get("status") == "validated"
            and validation.get("generation_count") == 5
            and repeat_ids == EXPECTED_REPEATS
            and not validation.get("errors")
            and all(row["valid"] for row in receipt_rows)
        )
        rows.append(
            {
                "case_id": case_id,
                "status": "validated" if valid else "invalid",
                "validation": str(validation_path),
                "generation_count": validation.get("generation_count"),
                "errors": validation.get("errors", []),
                "receipts": receipt_rows,
            }
        )
    return {
        "validated_case_count": sum(row["status"] == "validated" for row in rows),
        "blocked_case_count": sum(
            row["status"] == "blocked_by_declared_inputs_or_controls" for row in rows
        ),
        "excluded_case_count": sum(
            row["status"] == "excluded_by_execution_scope" for row in rows
        ),
        "invalid_or_missing_case_count": sum(
            row["status"] in {"invalid", "missing_validation"} for row in rows
        ),
        "cases": rows,
    }


def _matching_result_rows(
    path: Path,
    *,
    lite_case_id: str,
    dimension: str,
    evaluator_fingerprint: str,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reprocessing = row.get("mechanics_reprocessing") or {}
        row_fingerprint = (row.get("evaluator_lock") or {}).get(
            "evaluator_fingerprint"
        )
        fingerprint_matches = (
            row_fingerprint == evaluator_fingerprint
            or reprocessing.get("source_evaluator_fingerprint")
            == evaluator_fingerprint
        )
        if (
            row.get("case_id") != lite_case_id
            or row.get("dimension") != dimension
            or row.get("model") != "project_genie"
            or not fingerprint_matches
        ):
            continue
        selected[str(row.get("seed"))] = row
    return selected


def _result_artifacts_valid(
    *, case_id: str, results_path: Path, evaluator_fingerprint: str
) -> tuple[bool, dict[str, Any]]:
    if case_id.startswith("table4/"):
        path = results_path.parent / "diagnostic_summary.json"
        if not path.is_file():
            return False, {"diagnostic_summary": str(path), "artifact_status": "missing"}
        document = _read_json(path)
        result_rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        derived_rows = [row for row in result_rows if row.get("mechanics_reprocessing")]
        derived_valid = True
        if derived_rows:
            for row in derived_rows:
                provenance = row["mechanics_reprocessing"]
                source = Path(str(provenance.get("source_results", "")))
                if not source.is_file():
                    derived_valid = False
                    break
                source_bytes = source.read_bytes()
                source_line_hashes = {
                    hashlib.sha256(line.encode()).hexdigest()
                    for line in source.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
                if (
                    hashlib.sha256(source_bytes).hexdigest()
                    != provenance.get("source_results_sha256")
                    or provenance.get("source_row_sha256") not in source_line_hashes
                    or provenance.get("source_evaluator_fingerprint")
                    != evaluator_fingerprint
                    or (provenance.get("method") or {}).get("reporting_policy")
                    != "error_rate_only"
                    or "exact_match_pct" in (row.get("metrics") or {})
                ):
                    derived_valid = False
                    break
        valid = (
            document.get("status") == "metric_planar_calibrated_reconstructed"
            and document.get("generation_count") == 5
            and derived_valid
        )
        return valid, {
            "diagnostic_summary": str(path),
            "artifact_status": "validated" if valid else "invalid",
        }

    rerun_path = results_path.parent / "rerun_results.json"
    comparison_json = results_path.parent / "comparison" / "comparison.json"
    comparison_markdown = results_path.parent / "comparison" / "comparison.md"
    if not rerun_path.is_file():
        return False, {"rerun_results": str(rerun_path), "artifact_status": "missing"}
    document = _read_json(rerun_path)
    provenance = document.get("provenance") or {}
    incomplete_metrics = provenance.get("incomplete_metrics")
    recorded = (
        (provenance.get("evaluation_settings") or {})
        .get("evaluator_lock", {})
        .get("evaluator_fingerprint")
    )
    valid = (
        document.get("repeat_ids") == sorted(EXPECTED_REPEATS)
        and provenance.get("target_case_id") == case_id
        and recorded == evaluator_fingerprint
        and incomplete_metrics == []
        and comparison_json.is_file()
        and comparison_markdown.is_file()
    )
    if incomplete_metrics:
        artifact_status = "incomplete_metrics"
    else:
        artifact_status = "validated" if valid else "invalid"
    return valid, {
        "rerun_results": str(rerun_path),
        "comparison_json": str(comparison_json),
        "comparison_markdown": str(comparison_markdown),
        "incomplete_metrics": incomplete_metrics,
        "artifact_status": artifact_status,
    }


def _evaluation_status(generation: dict[str, Any] | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    generation_by_case = {
        str(row["case_id"]): row
        for row in (generation or {}).get("cases", [])
    }
    for case_id, reason in BLOCKED.items():
        rows.append({"case_id": case_id, "status": "blocked", "reason": reason})
    for case_id, reason in EXCLUDED.items():
        rows.append(
            {"case_id": case_id, "status": "excluded_by_user", "reason": reason}
        )

    for plan in PLANS:
        generation_row = generation_by_case.get(plan.case_id)
        if generation_row is not None and generation_row.get("status") != "validated":
            rows.append(
                {
                    "case_id": plan.case_id,
                    "status": "stale_or_incomplete_generation_inputs",
                    "generation_status": generation_row.get("status"),
                    "reason": (
                        "Existing evaluator outputs are not current because the five "
                        "generation receipts do not validate against the prepared inputs."
                    ),
                }
            )
            continue
        lite_case = load_case(
            REPO_ROOT / "lite" / "cases" / plan.lite_case_id
        )
        fingerprint = str(lite_case.evaluator_lock["evaluator_fingerprint"])
        table, scene, variant = plan.case_id.split("/")
        root = REPO_ROOT / "reports" / table / "project_genie" / scene / variant
        candidates: list[tuple[int, float, Path, dict[str, dict[str, Any]]]] = []
        ignored_stale_results: list[str] = []
        for path in root.glob("five_run_*/per_case_results.jsonl"):
            selected = _matching_result_rows(
                path,
                lite_case_id=plan.lite_case_id,
                dimension=plan.dimension,
                evaluator_fingerprint=fingerprint,
            )
            if selected:
                candidates.append((len(selected), path.stat().st_mtime, path, selected))
            else:
                ignored_stale_results.append(str(path))
        if not candidates:
            rows.append(
                {
                    "case_id": plan.case_id,
                    "status": "not_started",
                    "current_evaluator_fingerprint": fingerprint,
                    "completed_repeats": [],
                    "ignored_stale_results": ignored_stale_results,
                }
            )
            continue
        _, _, path, selected = max(candidates, key=lambda value: (value[0], value[1]))
        completed = {
            repeat_id
            for repeat_id, row in selected.items()
            if row.get("status") == "completed"
        }
        rows_complete = completed == EXPECTED_REPEATS and set(selected) == EXPECTED_REPEATS
        artifacts_valid, artifact_details = _result_artifacts_valid(
            case_id=plan.case_id,
            results_path=path,
            evaluator_fingerprint=fingerprint,
        )
        complete = rows_complete and artifacts_valid
        rows.append(
            {
                "case_id": plan.case_id,
                "status": (
                    "completed"
                    if complete
                    else (
                        "results_complete_metrics_incomplete"
                        if rows_complete
                        and artifact_details.get("artifact_status")
                        == "incomplete_metrics"
                        else (
                            "results_complete_artifacts_pending"
                            if rows_complete
                            else "in_progress"
                        )
                    )
                ),
                "results": str(path),
                "current_evaluator_fingerprint": fingerprint,
                "completed_repeats": sorted(completed),
                "missing_repeats": sorted(EXPECTED_REPEATS - completed),
                **artifact_details,
            }
        )
    runnable = [
        row
        for row in rows
        if row["status"] not in {"blocked", "excluded_by_user"}
    ]
    return {
        "runnable_case_count": len(runnable),
        "completed_case_count": sum(row["status"] == "completed" for row in runnable),
        "in_progress_case_count": sum(
            row["status"]
            in {
                "in_progress",
                "results_complete_artifacts_pending",
                "results_complete_metrics_incomplete",
                "stale_or_incomplete_generation_inputs",
            }
            for row in runnable
        ),
        "not_started_case_count": sum(row["status"] == "not_started" for row in runnable),
        "blocked_case_count": sum(row["status"] == "blocked" for row in rows),
        "excluded_case_count": sum(
            row["status"] == "excluded_by_user" for row in rows
        ),
        "cases": rows,
    }


def _queue_uses_calibrated_fixed_turns(queue_text: str) -> bool:
    table7_lines = [
        line for line in queue_text.splitlines() if "--case table7/" in line
    ]
    return len(table7_lines) == 12 and all(
        "--turn-calibration" in line and "--closed-loop-yaw" not in line
        for line in table7_lines
    )


def _browser_pipeline_status() -> dict[str, Any]:
    controller = REPO_ROOT / "tools" / "generation" / "project_genie_agent.py"
    queue_runner = ASSET_ROOT / "run_ready_project_genie.sh"
    full_runner = ASSET_ROOT / "run_full_pipeline.sh"
    controller_text = controller.read_text(encoding="utf-8") if controller.is_file() else ""
    queue_text = queue_runner.read_text(encoding="utf-8") if queue_runner.is_file() else ""
    checks = {
        "controller_exists": controller.is_file(),
        "queue_runner_executable": queue_runner.is_file()
        and bool(os.stat(queue_runner).st_mode & 0o111),
        "full_runner_executable": full_runner.is_file()
        and bool(os.stat(full_runner).st_mode & 0o111),
        "exact_policy_link_guard": 'name="Prohibited Use Policy", exact=True'
        in controller_text,
        "exact_i_agree_guard": 'name="I agree", exact=True' in controller_text,
        "queue_opts_into_policy_click": "--auto-accept-upload-policy" in queue_text,
        "queue_uses_resume": "--resume" in queue_text,
        "queue_uses_conditioning_overrides": "--first-frame-override" in queue_text,
        "queue_uses_calibrated_fixed_turns": _queue_uses_calibrated_fixed_turns(
            queue_text
        ),
    }
    receipts = sorted(
        (REPO_ROOT / "runs/project_genie").glob("table[47]/*/*/run[0-4]/receipt.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    live_evidence: dict[str, Any] | None = None
    for receipt_path in receipts:
        candidate = _validate_live_browser_receipt(receipt_path)
        if candidate["status"] == "validated":
            live_evidence = candidate
            break
    checks["validated_live_end_to_end_receipt"] = live_evidence is not None
    return {
        "status": "ready" if all(checks.values()) else "failed",
        "checks": checks,
        "controller": str(controller),
        "queue_runner": str(queue_runner),
        "full_runner": str(full_runner),
        "live_evidence": live_evidence,
    }


def _validate_live_browser_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "receipt": str(path), "error": str(exc)}

    def hashed_file(path_key: str, hash_key: str) -> bool:
        value = receipt.get(path_key)
        expected = receipt.get(hash_key)
        if not isinstance(value, str) or not isinstance(expected, str):
            return False
        artifact = Path(value)
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        return artifact.is_file() and _sha256(artifact) == expected

    acceptances = receipt.get("upload_policy_acceptances") or []
    exact_consent = any(
        isinstance(item, dict)
        and item.get("button_label") == "I agree"
        and item.get("policy_label") == "Prohibited Use Policy"
        and item.get("mode") == "automated_explicit_cli_opt_in"
        for item in acceptances
    )
    screenshots = receipt.get("screenshots") or {}
    required_screenshots = {
        "frame_uploaded",
        "prompts_ready",
        "generation_requested",
        "world_ready",
        "actions_complete",
        "complete",
    }
    checks = {
        "complete": receipt.get("status") == "complete",
        "independent_generation": receipt.get("generation_mode") == "new_world_generation"
        and receipt.get("table_repeat_eligible") is True,
        "conditioning_frame_hashed": hashed_file("first_frame", "first_frame_sha256"),
        "actions_hashed": hashed_file("actions", "actions_sha256"),
        "accepted_action_trace_hashed": hashed_file(
            "action_trace", "action_trace_sha256"
        ),
        "evaluation_video_hashed": hashed_file(
            "evaluation_video", "evaluation_video_sha256"
        ),
        "native_alignment_hashed": hashed_file(
            "native_alignment_report", "native_alignment_report_sha256"
        ),
        "native_timeline_validated": receipt.get("native_video_timeline_alignment")
        == "validated_visual_correspondence",
        "prompt_fields_recorded": isinstance(receipt.get("environment_prompt"), str)
        and bool(receipt.get("environment_prompt"))
        and isinstance(receipt.get("character_prompt"), str),
        "view_recorded": receipt.get("view") in {"first_person", "third_person"},
        "checkpoint_screenshots_recorded": required_screenshots.issubset(screenshots),
        "exact_upload_policy_consent_recorded": exact_consent,
    }
    return {
        "status": "validated" if all(checks.values()) else "invalid",
        "receipt": str(path),
        "checks": checks,
        "case_id": str(path.parent.parent.relative_to(REPO_ROOT / "runs/project_genie")),
        "repeat_id": path.parent.name,
        "evaluation_video_sha256": receipt.get("evaluation_video_sha256"),
    }


def _metric_source_readiness() -> dict[str, Any]:
    """Audit source artifacts that impose metric-level, not case-level, blockers."""
    mirror_transforms = (
        REPO_ROOT
        / "lite/cases/MirrorRollingBall/prompt_third"
        / "reference/artifacts/object_transforms.json"
    )
    object_labels: list[str] = []
    error: str | None = None
    if not mirror_transforms.is_file():
        error = "Unity object-transform artifact is missing"
    else:
        try:
            objects = _read_json(mirror_transforms).get("objects")
            if not isinstance(objects, dict):
                error = "Unity object-transform artifact has no objects mapping"
            else:
                object_labels = sorted(str(label) for label in objects)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
    pairwise_ready = error is None and len(object_labels) >= 2
    mirror_row = {
        "metric": "table7/MirrorRollingBall/*/dist_mae_pct",
        "status": (
            "ready"
            if pairwise_ready
            else "blocked_insufficient_unique_unity_objects"
        ),
        "source": str(mirror_transforms),
        "unique_object_count": len(object_labels),
        "object_labels": object_labels,
        "minimum_required": 2,
        "reason": (
            error
            or (
                "Pairwise-distance MAE has two fixed, uniquely identified "
                "foreground sphere anchors separated by 4.0 Unity metres."
                if pairwise_ready
                else "Pairwise-distance MAE requires at least two uniquely "
                "identified Unity objects."
            )
        ),
    }
    return {
        "status": "ready" if pairwise_ready else "blocked",
        "metrics": [mirror_row],
    }


def build_status() -> dict[str, Any]:
    paper = _paper_sources()
    table_audit = audit_table_trajectories(TABLE_ROOT, strict=False)
    strict_audit = audit_table_trajectories(TABLE_ROOT, strict=True)
    trajectory_inventory = inventory(
        REPO_ROOT / "lite" / "cases"
    )
    generation = _generation_status()
    evaluation = _evaluation_status(generation)
    browser = _browser_pipeline_status()
    metric_sources = _metric_source_readiness()
    trajectory_ready = trajectory_inventory.get("effective_status_counts", {}) == {
        "native_machine_readable_ready": 15,
        "supplemental_reconstruction_ready": 4,
    }
    reconstruction_ready = (
        paper["status"] == "verified"
        and table_audit["ok"]
        and trajectory_ready
        and len(generation["cases"]) == 16
        and generation["blocked_case_count"] == 0
        and generation["excluded_case_count"] == len(EXCLUDED)
        and browser["status"] == "ready"
        and metric_sources["status"] == "ready"
    )
    execution_complete = (
        reconstruction_ready
        and generation["invalid_or_missing_case_count"] == 0
        and evaluation["completed_case_count"] == evaluation["runnable_case_count"]
    )
    return {
        "schema_version": 1,
        "claim_scope": "paper_protocol_reconstruction_not_exact_paper_parity",
        "status": "complete" if execution_complete else "in_progress",
        "reconstruction_assets_status": "ready" if reconstruction_ready else "failed",
        "exact_paper_parity_status": (
            "verified" if strict_audit["ok"] else "blocked_by_recorded_source_gaps"
        ),
        "paper_sources": paper,
        "table_protocol_audit": table_audit,
        "strict_paper_parity_audit": strict_audit,
        "trajectory_inventory": {
            "case_count": trajectory_inventory["case_count"],
            "effective_status_counts": trajectory_inventory[
                "effective_status_counts"
            ],
            "status": "ready" if trajectory_ready else "failed",
        },
        "generation": generation,
        "browser_pipeline": browser,
        "metric_source_readiness": metric_sources,
        "evaluation": evaluation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "runs" / "tables_4_7" / "reproduction_status.json",
    )
    args = parser.parse_args()
    result = build_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["reconstruction_assets_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
