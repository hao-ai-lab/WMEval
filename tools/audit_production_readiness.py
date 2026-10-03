#!/usr/bin/env python3
"""Audit cross-dimension production readiness for WMEval Lite.

The suite validator answers whether case bundles are internally valid. This
audit answers the separate operational question: which dimensions have native
trajectories, portable backend controls, complete evaluator provenance, and a
documented remediation for known gaps?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_trajectory_readiness import (
    DEFAULT_CASES,
    DEFAULT_OTHER_ROOT,
    DEFAULT_SUITE,
    DEFAULT_TABLE_ROOT,
    inventory,
)

DEFAULT_STANDARDIZATION = REPO_ROOT / "experiments" / "evaluation_dimension_standardization.yaml"
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}
VALID_READINESS = {"ready", "partial", "blocked"}


def _mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _normalization(case_ids: list[str], cases_root: Path) -> list[str]:
    policies: set[str] = set()
    for case_id in case_ids:
        case = _mapping(cases_root / case_id / "case.yaml")
        protocol = case.get("protocol") or {}
        policies.add(str(protocol.get("candidate_normalization", "missing")))
    return sorted(policies)


def _lock_coverage(case_ids: list[str], cases_root: Path) -> dict[str, int]:
    keys = ("module_config", "artifact_schema_versions", "dependency_versions")
    counts = {key: 0 for key in keys}
    for case_id in case_ids:
        case = _mapping(cases_root / case_id / "case.yaml")
        lock = case.get("evaluator_lock") or {}
        for key in keys:
            if lock.get(key):
                counts[key] += 1
    return counts


def _gap_counts(gaps: list[dict[str, Any]]) -> dict[str, int]:
    counts = {severity: 0 for severity in SEVERITY_ORDER}
    for gap in gaps:
        severity = str(gap.get("severity", ""))
        if severity in counts and gap.get("status", "open") != "closed":
            counts[severity] += 1
    return counts


def _priority_counts(gaps: list[dict[str, Any]]) -> dict[str, int]:
    counts = {priority: 0 for priority in PRIORITY_ORDER}
    for gap in gaps:
        priority = str(gap.get("priority", ""))
        if priority in counts and gap.get("status", "open") != "closed":
            counts[priority] += 1
    return counts


def _gap_sort_key(gap: dict[str, Any]) -> tuple[int, int, str]:
    return (
        PRIORITY_ORDER.get(str(gap.get("priority")), 99),
        SEVERITY_ORDER.get(str(gap.get("severity")), 99),
        str(gap.get("id", gap.get("gap", ""))),
    )


def build_report(
    *,
    suite_path: Path = DEFAULT_SUITE,
    cases_root: Path = DEFAULT_CASES,
    standardization_path: Path = DEFAULT_STANDARDIZATION,
    table_root: Path = DEFAULT_TABLE_ROOT,
    other_root: Path = DEFAULT_OTHER_ROOT,
) -> dict[str, Any]:
    suite = _mapping(suite_path)
    standardization = _mapping(standardization_path)
    dimensions = suite.get("dimensions") or {}
    declared = standardization.get("dimensions") or {}
    errors: list[str] = []
    if standardization.get("schema_version") != 2:
        errors.append("standardization schema_version must be 2")
    missing = sorted(set(dimensions) - set(declared))
    extra = sorted(set(declared) - set(dimensions))
    if missing:
        errors.append(f"dimensions missing from standardization manifest: {missing}")
    if extra:
        errors.append(f"unknown dimensions in standardization manifest: {extra}")

    trajectory = inventory(
        cases_root,
        suite_path=suite_path,
        table_root=table_root,
        other_root=other_root,
    )
    trajectory_by_case = {row["case_id"]: row for row in trajectory["cases"]}
    rows: list[dict[str, Any]] = []
    for dimension, config in dimensions.items():
        case_ids = [str(case_id) for case_id in (config.get("cases") or [])]
        contract = declared.get(dimension) or {}
        readiness = str(contract.get("readiness", "missing"))
        if readiness not in VALID_READINESS:
            errors.append(f"{dimension}: invalid readiness {readiness!r}")
        gaps = contract.get("gaps") or []
        for index, gap in enumerate(gaps):
            if not isinstance(gap, dict):
                errors.append(f"{dimension}.gaps[{index}] must be a mapping")
                continue
            severity = gap.get("severity")
            if severity not in SEVERITY_ORDER:
                errors.append(f"{dimension}.gaps[{index}] has invalid severity {severity!r}")
            priority = gap.get("priority")
            if priority not in PRIORITY_ORDER:
                errors.append(f"{dimension}.gaps[{index}] has invalid priority {priority!r}")
            if not str(gap.get("gap", "")).strip() or not str(gap.get("remediation", "")).strip():
                errors.append(f"{dimension}.gaps[{index}] needs gap and remediation text")

        case_rows = [trajectory_by_case[case_id] for case_id in case_ids]
        native = sum(row["effective_status"] == "native_machine_readable_ready" for row in case_rows)
        supplemental = sum(row["effective_status"] == "supplemental_reconstruction_ready" for row in case_rows)
        browser_ready = sum(
            row["backend_readiness"]["project_genie_browser"]
            in {"native_executable_ready", "supplemental_executable_ready", "mixed_ready_and_unsupported"}
            for row in case_rows
        )
        rows.append(
            {
                "dimension": dimension,
                "readiness": readiness,
                "case_count": len(case_ids),
                "cases": case_ids,
                "trajectory_coverage": {
                    "native": native,
                    "supplemental": supplemental,
                    "other": len(case_ids) - native - supplemental,
                },
                "project_genie_executable_cases": browser_ready,
                "normalization_policies": _normalization(case_ids, cases_root),
                "evaluator_lock_nonempty_case_counts": _lock_coverage(case_ids, cases_root),
                "standardized": contract.get("standardized") or {},
                "gap_counts": _gap_counts(gaps),
                "priority_counts": _priority_counts(gaps),
                "gaps": gaps,
            }
        )

    cross_cutting = standardization.get("cross_cutting_gaps") or []
    for index, gap in enumerate(cross_cutting):
        if not isinstance(gap, dict):
            errors.append(f"cross_cutting_gaps[{index}] must be a mapping")
            continue
        if gap.get("severity") not in SEVERITY_ORDER:
            errors.append(f"cross_cutting_gaps[{index}] has invalid severity")
        if gap.get("priority") not in PRIORITY_ORDER:
            errors.append(f"cross_cutting_gaps[{index}] has invalid priority")
        for key in ("id", "gap", "remediation"):
            if not str(gap.get(key, "")).strip():
                errors.append(f"cross_cutting_gaps[{index}] needs {key}")

    all_gaps = list(cross_cutting)
    for row in rows:
        all_gaps.extend(row["gaps"])
    return {
        "schema_version": 1,
        "status": "valid" if not errors else "invalid",
        "suite": str(suite_path),
        "standardization_manifest": str(standardization_path),
        "summary": {
            "dimension_count": len(rows),
            "case_count": trajectory["case_count"],
            "native_trajectory_case_count": trajectory["effective_status_counts"].get(
                "native_machine_readable_ready", 0
            ),
            "supplemental_trajectory_case_count": trajectory["effective_status_counts"].get(
                "supplemental_reconstruction_ready", 0
            ),
            "project_genie_backend_readiness": trajectory["backend_readiness_counts"][
                "project_genie_browser"
            ],
            "open_gap_counts": _gap_counts(all_gaps),
            "open_priority_counts": _priority_counts(all_gaps),
        },
        "manifest_errors": errors,
        "cross_cutting_gaps": sorted(cross_cutting, key=_gap_sort_key),
        "dimensions": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# WMEval production-readiness audit",
        "",
        "Generated from the live suite, case bundles, trajectory inventories, and "
        "`experiments/evaluation_dimension_standardization.yaml`.",
        "",
        "A valid case bundle is not automatically a portable or paper-parity experiment. "
        "The table below keeps those concepts separate.",
        "",
        "## Snapshot",
        "",
        f"- {summary['case_count']} validated case bundles across {summary['dimension_count']} dimensions.",
        f"- {summary['native_trajectory_case_count']} cases have native machine-readable trajectories; "
        f"{summary['supplemental_trajectory_case_count']} rely on supplemental reconstructions.",
        f"- Project Genie backend readiness: `{json.dumps(summary['project_genie_backend_readiness'], sort_keys=True)}`.",
        f"- Open gaps by severity: `{json.dumps(summary['open_gap_counts'], sort_keys=True)}`.",
        f"- Open gaps by release priority: `{json.dumps(summary['open_priority_counts'], sort_keys=True)}`.",
        "",
        "## Dimension comparison",
        "",
        "| Dimension | Readiness | Cases | Native / supplemental trajectories | Genie-executable | Normalization | P0 / P1 / P2 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in report["dimensions"]:
        coverage = row["trajectory_coverage"]
        policies = ", ".join(f"`{value}`" for value in row["normalization_policies"])
        priorities = row["priority_counts"]
        lines.append(
            f"| `{row['dimension']}` | {row['readiness']} | {row['case_count']} | "
            f"{coverage['native']} / {coverage['supplemental']} | "
            f"{row['project_genie_executable_cases']}/{row['case_count']} | {policies} | "
            f"{priorities['P0']} / {priorities['P1']} / {priorities['P2']} |"
        )

    lines.extend(["", "## Cross-cutting backlog", ""])
    for gap in report["cross_cutting_gaps"]:
        lines.extend(
            [
                f"- **{gap['priority']} / {str(gap['severity']).upper()} — `{gap['id']}`:** {gap['gap']}",
                f"  Remediation: {gap['remediation']}",
            ]
        )

    lines.extend(["", "## Dimension-specific backlog", ""])
    for row in report["dimensions"]:
        lines.append(f"### {row['dimension']}")
        lines.append("")
        for gap in sorted(
            (item for item in row["gaps"] if item.get("status", "open") != "closed"),
            key=_gap_sort_key,
        ):
            lines.extend(
                [
                    f"- **{gap['priority']} / {str(gap['severity']).upper()}:** {gap['gap']}",
                    f"  Remediation: {gap['remediation']}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Reproduce",
            "",
            "```bash",
            "uv run python tools/run_wmeval_lite.py validate --suite eval_sets/wmeval_lite_v1",
            "uv run python tools/audit_production_readiness.py --markdown PRODUCTION_READINESS.md --json reports/production_readiness.json",
            "```",
            "",
            "`--fail-on-high` is intended as a future release gate; it currently fails because the "
            "documented high-severity backlog is still open.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--standardization", type=Path, default=DEFAULT_STANDARDIZATION)
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--other-root", type=Path, default=DEFAULT_OTHER_ROOT)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--fail-on-high", action="store_true")
    args = parser.parse_args()
    report = build_report(
        suite_path=args.suite.resolve(),
        cases_root=args.cases_root.resolve(),
        standardization_path=args.standardization.resolve(),
        table_root=args.table_root.resolve(),
        other_root=args.other_root.resolve(),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded, encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    if not args.json and not args.markdown:
        print(encoded, end="")
    if report["manifest_errors"]:
        return 2
    if args.fail_on_high:
        counts = report["summary"]["open_gap_counts"]
        if counts["critical"] or counts["high"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
