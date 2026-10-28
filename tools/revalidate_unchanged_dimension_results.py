#!/usr/bin/env python3
"""Revalidate stored raw reports when an unrelated global fingerprint changed.

This is deliberately stricter than editing a fingerprint in place.  It only
accepts completed rows whose dimension module source predates the source
result, recomputes the row's flattened metrics from the stored full report
with the current extractor, requires exact equality, and records cryptographic
provenance for the migration.  It is useful because the v1 case lock hashes
all evaluator sources, so a kinematics-only edit otherwise invalidates costly
environment-consistency reports even when that module and its outputs did not
change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.lite.aggregate import (
    build_aggregate,
    extract_metrics,
    write_aggregate_json,
    write_summary_csv,
)
from evaluation.lite.case_loader import load_case
from evaluation.lite.output import read_rows, write_rows
from evaluation.lite.suite_loader import load_suite
from tools.run_prepared_table_evaluations import (
    PLANS,
    _write_comparison_artifacts,
    refresh_consolidated_comparison,
)


MODULE_SOURCES = {
    "environment_object_consistency": (
        REPO_ROOT / "evaluation/modules/environment_object_consistency.py"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revalidate_rows(
    *, source_results: Path, target_case_id: str, prepared_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plans = {plan.case_id: plan for plan in PLANS}
    plan = plans.get(target_case_id)
    if plan is None or not target_case_id.startswith("table7/"):
        raise ValueError(f"unsupported Table 7 target: {target_case_id}")
    suite = load_suite(REPO_ROOT / "lite")
    dimension = suite.dimensions[plan.dimension]
    if tuple(dimension.modules) != ("environment_object_consistency",):
        raise ValueError("revalidation is limited to the unchanged EOC dimension")
    module_source = MODULE_SOURCES["environment_object_consistency"]
    if module_source.stat().st_mtime > source_results.stat().st_mtime:
        raise ValueError(
            "EOC module source is newer than the stored result; a real rerun is required"
        )
    case = load_case(REPO_ROOT / "lite/cases" / plan.lite_case_id)
    inputs = json.loads((prepared_dir / "evaluation_inputs.json").read_text())
    current_candidates = {
        str(item["repeat_id"]): str(Path(item["candidate_video"]).resolve())
        for item in inputs["runs"]
    }
    rows = read_rows(source_results)
    expected = {f"run{i}" for i in range(5)}
    if {str(row.get("seed")) for row in rows} != expected:
        raise ValueError("source result must contain exactly run0 through run4")
    migrated: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item.get("seed"))):
        seed = str(row.get("seed"))
        if (
            row.get("status") != "completed"
            or row.get("case_id") != plan.lite_case_id
            or row.get("dimension") != plan.dimension
            or row.get("model") != "project_genie"
        ):
            raise ValueError(f"{seed}: source row identity/status mismatch")
        if str(Path(str(row.get("candidate_video"))).resolve()) != current_candidates[seed]:
            raise ValueError(f"{seed}: candidate path differs from current prepared input")
        full_report = row.get("full_report")
        results = full_report.get("results") if isinstance(full_report, dict) else None
        if not isinstance(results, dict):
            raise ValueError(f"{seed}: stored raw module report is missing")
        recomputed = extract_metrics(dimension, results)
        if recomputed != row.get("metrics"):
            raise ValueError(
                f"{seed}: current metric extraction differs from stored metrics; rerun required"
            )
        updated = dict(row)
        updated["evaluator_lock"] = {
            "release_id": case.evaluator_lock.get("release_id"),
            "evaluator_fingerprint": case.evaluator_lock.get("evaluator_fingerprint"),
        }
        updated["revalidation"] = {
            "mode": "stored_raw_report_reextracted_under_unchanged_dimension_module",
            "source_results": str(source_results.resolve()),
            "source_results_sha256": sha256(source_results),
            "dimension_module_source": str(module_source),
            "dimension_module_source_sha256": sha256(module_source),
            "module_source_mtime_not_after_source_results": True,
            "current_metric_extraction_exact_match": True,
            "candidate_path_matches_current_prepared_input": True,
        }
        migrated.append(updated)
    evidence = {
        "schema_version": 1,
        "status": "revalidated",
        "target_case_id": target_case_id,
        "source_results": str(source_results.resolve()),
        "source_results_sha256": sha256(source_results),
        "source_results_mtime": source_results.stat().st_mtime,
        "dimension_module_source": str(module_source),
        "dimension_module_source_sha256": sha256(module_source),
        "dimension_module_source_mtime": module_source.stat().st_mtime,
        "row_count": len(migrated),
        "repeat_ids": sorted(expected),
        "metric_extraction_exact_match": True,
        "claim_scope": "unchanged_dimension_raw_report_revalidation",
    }
    return migrated, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--target-case-id", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_results.resolve()
    prepared = args.prepared_dir.resolve()
    output = args.output_dir.resolve()
    rows, evidence = revalidate_rows(
        source_results=source,
        target_case_id=args.target_case_id,
        prepared_dir=prepared,
    )
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "per_case_results.jsonl"
    write_rows(results_path, rows)
    suite = load_suite(REPO_ROOT / "lite")
    aggregate = build_aggregate(suite, rows)
    write_aggregate_json(output / "aggregate_summary.json", aggregate)
    write_summary_csv(output / "aggregate_summary.csv", suite, aggregate)
    evidence["results"] = str(results_path)
    (output / "revalidation.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan = next(plan for plan in PLANS if plan.case_id == args.target_case_id)
    evidence.update(_write_comparison_artifacts(plan, prepared, output))
    evidence.update(refresh_consolidated_comparison())
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
