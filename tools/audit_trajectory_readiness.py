#!/usr/bin/env python3
"""Inventory whether every WMEval Lite case has usable trajectory assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "lite" / "cases"
DEFAULT_SUITE = REPO_ROOT / "lite" / "suite.yaml"
DEFAULT_TABLE_ROOT = REPO_ROOT / "experiments" / "protocol"
DEFAULT_OTHER_ROOT = REPO_ROOT / "experiments" / "trajectory_reconstructions"


def parse_json_ops(text: str) -> list[dict[str, Any]]:
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return []
    body = "\n".join(lines)
    try:
        value = json.loads(body)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict) and item.get("op")]
    except json.JSONDecodeError:
        pass
    ops: list[dict[str, Any]] = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            return []
        if isinstance(item, dict) and item.get("op"):
            ops.append(item)
    return ops


def inspect_case(
    case_dir: Path,
    repo_root: Path,
    *,
    dimensions: list[str] | None = None,
    supplemental_trajectories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    artifacts = case_dir / "reference" / "artifacts"
    actions_txt = artifacts / "actions.txt"
    action_npy = artifacts / "action.npy"
    trajectory_path = artifacts / "trajectory.yaml"
    trajectory = yaml.safe_load(trajectory_path.read_text()) if trajectory_path.is_file() else {}
    text_kind = "missing"
    op_count = 0
    if actions_txt.is_file():
        text = actions_txt.read_text(encoding="utf-8")
        ops = parse_json_ops(text)
        op_count = len(ops)
        non_comment = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        text_kind = "executable_json" if ops else ("structured_text" if non_comment else "comment_only")
    # Compiled Table 4/7 bundles keep the numeric schedule in actions.txt and the
    # Project Genie browser program in actions.jsonl.
    actions_jsonl = artifacts / "actions.jsonl"
    if not op_count and actions_jsonl.is_file():
        op_count = len(parse_json_ops(actions_jsonl.read_text(encoding="utf-8")))

    numeric: dict[str, Any] | None = None
    if action_npy.is_file():
        array = np.load(action_npy, allow_pickle=False)
        numeric = {"shape": list(array.shape), "dtype": str(array.dtype)}
    action_npz = artifacts / "action.npz"
    if numeric is None and action_npz.is_file():
        # Standardized bundles may carry only the compiled semantic trajectory.
        with np.load(action_npz, allow_pickle=False) as compiled:
            values = compiled["values"]
            numeric = {"shape": list(values.shape), "dtype": str(values.dtype), "source": "action.npz"}

    if numeric and (text_kind in {"executable_json", "structured_text"} or trajectory):
        status = "ready"
    elif numeric:
        status = "numeric_only"
    elif text_kind == "executable_json":
        status = "executable_script_only"
    elif text_kind == "structured_text":
        status = "semantic_text_only"
    else:
        status = "underspecified"

    supplements = supplemental_trajectories or []
    if status == "ready":
        effective_status = "native_machine_readable_ready"
        issues: list[str] = []
    elif supplements:
        effective_status = "supplemental_reconstruction_ready"
        issues = [
            "the locked Lite case lacks native controls; reconstructed controls live outside the case bundle"
        ]
    elif status == "executable_script_only":
        effective_status = "browser_ready_numeric_missing"
        issues = ["numeric action.npy is missing"]
    elif text_kind == "comment_only":
        effective_status = "documented_schedule_not_machine_readable"
        issues = ["the capture schedule is documented only in comments"]
    else:
        effective_status = "underspecified"
        issues = ["no machine-readable or supplemental trajectory is available"]

    supplemental_numeric = any(item.get("compiled_numeric") for item in supplements)
    supplemental_browser = any(
        item.get("project_genie_status") not in {None, "unsupported"}
        and item.get("project_genie_actions")
        for item in supplements
    )
    explicitly_unsupported = any(
        item.get("project_genie_status") == "unsupported" for item in supplements
    )
    registry = repo_root / 'experiments/protocol/action_backends.yaml'
    if trajectory and registry.is_file():
        support = yaml.safe_load(registry.read_text()).get('trajectory_support', {}).get(
            trajectory.get('trajectory_id'), {}).get('backends', {})
        explicitly_unsupported |= support.get('project_genie', {}).get('status') == 'unsupported'
    if numeric is not None:
        unity_readiness = "native_numeric_ready"
    elif supplemental_numeric:
        unity_statuses = sorted(
            {str(item.get("unity_status", "unspecified")) for item in supplements}
        )
        unity_readiness = "supplemental_numeric_ready"
        if unity_statuses == ["semantic_only"]:
            unity_readiness = "supplemental_semantic_numeric_only"
    else:
        unity_readiness = "missing_numeric_trajectory"

    if op_count:
        project_genie_readiness = "native_executable_ready"
    elif supplemental_browser and explicitly_unsupported:
        project_genie_readiness = "mixed_ready_and_unsupported"
    elif supplemental_browser:
        project_genie_readiness = "supplemental_executable_ready"
    elif explicitly_unsupported:
        project_genie_readiness = "unsupported_action_space"
    else:
        project_genie_readiness = "not_machine_executable"

    return {
        "case_id": f"{case_dir.parent.name}/{case_dir.name}",
        "dimensions": dimensions or [],
        "status": status,
        "effective_status": effective_status,
        "issues": issues,
        "actions_text_kind": text_kind,
        "browser_op_count": op_count,
        "numeric_action": numeric,
        "actions_file": str(actions_txt.relative_to(repo_root)) if actions_txt.is_file() else None,
        "numeric_file": str(action_npy.relative_to(repo_root)) if action_npy.is_file() else None,
        "supplemental_trajectories": supplements,
        "backend_readiness": {
            "unity_or_numeric_evaluator": unity_readiness,
            "project_genie_browser": project_genie_readiness,
        },
    }


def _dimension_map(suite_path: Path) -> dict[str, list[str]]:
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for dimension, config in suite.get("dimensions", {}).items():
        for case_id in config.get("cases", []):
            result.setdefault(str(case_id), []).append(str(dimension))
    return result


def _supplemental_map(table_root: Path, repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    manifest_path = table_root / "cases.yaml"
    if not manifest_path.is_file():
        return {}
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for case in manifest.get("cases", []):
        prompt_source = (table_root / str(case.get("prompt_source", ""))).resolve()
        trajectory = (table_root / str(case.get("trajectory", ""))).resolve()
        if not prompt_source.is_file() or not trajectory.is_file():
            continue
        case_doc = yaml.safe_load(prompt_source.read_text(encoding="utf-8"))
        lite_case_id = str(case_doc.get("case_id", ""))
        key = (lite_case_id, str(trajectory))
        if not lite_case_id or key in seen:
            continue
        seen.add(key)
        trajectory_doc = yaml.safe_load(trajectory.read_text(encoding="utf-8"))
        compiled = table_root / "compiled" / f"{trajectory.stem}.action.npy"
        compiled_manifest = table_root / "compiled" / f"{trajectory.stem}.manifest.json"
        compiled_doc = (
            json.loads(compiled_manifest.read_text(encoding="utf-8"))
            if compiled_manifest.is_file()
            else {}
        )
        browser_actions = table_root / "compiled" / str(
            compiled_doc.get("project_genie_actions", "")
        )
        result.setdefault(lite_case_id, []).append(
            {
                "trajectory_id": trajectory_doc.get("trajectory_id"),
                "paper_alignment_status": trajectory_doc.get("paper_alignment", {}).get(
                    "status"
                ),
                "source": str(trajectory.relative_to(repo_root)),
                "compiled_numeric": (
                    str(compiled.relative_to(repo_root)) if compiled.is_file() else None
                ),
                "unity_status": trajectory_doc.get("adapters", {})
                .get("unity", {})
                .get("status"),
                "project_genie_status": compiled_doc.get("project_genie_status"),
                "project_genie_actions": (
                    str(browser_actions.relative_to(repo_root))
                    if browser_actions.is_file() and browser_actions.stat().st_size > 0
                    else None
                ),
            }
        )
    return result


def _other_supplemental_map(
    root: Path, repo_root: Path
) -> dict[str, list[dict[str, Any]]]:
    manifest_path = root / "manifest.yaml"
    if not manifest_path.is_file():
        return {}
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    for case in manifest.get("cases", []):
        case_id = str(case.get("case_id", ""))
        trajectory = (root / str(case.get("trajectory", ""))).resolve()
        if not case_id or not trajectory.is_file():
            continue
        trajectory_doc = yaml.safe_load(trajectory.read_text(encoding="utf-8"))
        compiled = root / "compiled" / f"{trajectory.stem}.action.npy"
        compiled_manifest = root / "compiled" / f"{trajectory.stem}.manifest.json"
        compiled_doc = (
            json.loads(compiled_manifest.read_text(encoding="utf-8"))
            if compiled_manifest.is_file()
            else {}
        )
        browser_actions = root / "compiled" / str(
            compiled_doc.get("project_genie_actions", "")
        )
        result.setdefault(case_id, []).append(
            {
                "trajectory_id": trajectory_doc.get("trajectory_id"),
                "paper_alignment_status": trajectory_doc.get("paper_alignment", {}).get(
                    "status"
                ),
                "source": str(trajectory.relative_to(repo_root)),
                "compiled_numeric": (
                    str(compiled.relative_to(repo_root)) if compiled.is_file() else None
                ),
                "unity_status": trajectory_doc.get("adapters", {})
                .get("unity", {})
                .get("status"),
                "project_genie_status": compiled_doc.get("project_genie_status"),
                "project_genie_actions": (
                    str(browser_actions.relative_to(repo_root))
                    if browser_actions.is_file() and browser_actions.stat().st_size > 0
                    else None
                ),
            }
        )
    return result


def inventory(
    cases_root: Path,
    repo_root: Path = REPO_ROOT,
    *,
    suite_path: Path = DEFAULT_SUITE,
    table_root: Path = DEFAULT_TABLE_ROOT,
    other_root: Path = DEFAULT_OTHER_ROOT,
) -> dict[str, Any]:
    dimensions = _dimension_map(suite_path)
    supplements = _supplemental_map(table_root, repo_root)
    for case_id, rows in _other_supplemental_map(other_root, repo_root).items():
        supplements.setdefault(case_id, []).extend(rows)
    rows = [
        inspect_case(
            case_yaml.parent,
            repo_root,
            dimensions=dimensions.get(
                f"{case_yaml.parent.parent.name}/{case_yaml.parent.name}", []
            ),
            supplemental_trajectories=supplements.get(
                f"{case_yaml.parent.parent.name}/{case_yaml.parent.name}", []
            ),
        )
        for case_yaml in sorted(cases_root.glob("*/*/case.yaml"))
    ]
    counts: dict[str, int] = {}
    effective_counts: dict[str, int] = {}
    backend_counts: dict[str, dict[str, int]] = {
        "unity_or_numeric_evaluator": {},
        "project_genie_browser": {},
    }
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        effective = row["effective_status"]
        effective_counts[effective] = effective_counts.get(effective, 0) + 1
        for backend, status in row["backend_readiness"].items():
            backend_counts[backend][status] = backend_counts[backend].get(status, 0) + 1
    return {
        "schema_version": 3,
        "case_count": len(rows),
        "status_counts": counts,
        "effective_status_counts": effective_counts,
        "backend_readiness_counts": backend_counts,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--other-root", type=Path, default=DEFAULT_OTHER_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inventory(
        args.cases_root.resolve(),
        suite_path=args.suite.resolve(),
        table_root=args.table_root.resolve(),
        other_root=args.other_root.resolve(),
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
