#!/usr/bin/env python3
"""Validate and compile the canonical Table 4/Table 7 trajectories."""
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

from evaluation.trajectories import compile_trajectory, write_compiled_trajectory
from evaluation.table_comparison import load_baseline

DEFAULT_ROOT = REPO_ROOT / "experiments" / "tables_4_7"


def load_cases(root: Path) -> dict[str, Any]:
    manifest = root / "cases.yaml"
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"{manifest}: unsupported manifest")
    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{manifest}: cases must be a list")
    ids = [case.get("case_id") for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{manifest}: duplicate case_id")
    return raw


def audit(root: Path, *, strict: bool) -> dict[str, Any]:
    manifest = load_cases(root)
    cases = manifest["cases"]
    errors: list[str] = []
    warnings: list[str] = []
    expected = {4: 4, 7: 12}
    counts = {table: sum(case.get("table") == table for case in cases) for table in expected}
    for table, count in expected.items():
        if counts[table] != count:
            errors.append(f"Table {table}: expected {count} cells, found {counts[table]}")

    baseline_path = (root / str(manifest.get("paper_baseline", "paper_baseline.yaml"))).resolve()
    try:
        baseline = load_baseline(baseline_path)
        baseline_ids = {
            case_id
            for table in expected
            for case_id in baseline["tables"][table]["cases"]
        }
        manifest_ids = {str(case["case_id"]) for case in cases}
        if baseline_ids != manifest_ids:
            errors.append(
                "case manifest differs from paper baseline: "
                f"missing={sorted(baseline_ids - manifest_ids)}, extra={sorted(manifest_ids - baseline_ids)}"
            )
        if manifest.get("generation_repeats") != baseline["aggregation"]["generation_count"]:
            errors.append("generation repeat count differs from paper baseline")
        if manifest.get("repeat_ids") != [f"run{index}" for index in range(5)]:
            errors.append("repeat_ids must be run0 through run4")
        for conflict in baseline.get("paper_conflicts", []):
            message = f"paper conflict {conflict['id']}: {conflict['status']}"
            (errors if strict and conflict.get("status") == "unresolved" else warnings).append(message)
    except Exception as exc:
        errors.append(f"paper baseline validation failed: {exc}")

    prompt_components = (root / str(manifest.get("prompt_components", "prompt_components.yaml"))).resolve()
    try:
        prompt_raw = yaml.safe_load(prompt_components.read_text(encoding="utf-8"))
        if prompt_raw.get("schema_version") != 1:
            raise ValueError("schema_version must be 1")
        for key, expected_keys in {
            "scenes": {"WanderingGarden", "BilliardGame", "MirrorRollingBall"},
            "orientations": {"inward", "outward"},
            "motions": {"perturb", "360"},
        }.items():
            missing_keys = expected_keys - set(prompt_raw.get(key, {}))
            if missing_keys:
                raise ValueError(f"{key} missing {sorted(missing_keys)}")
    except Exception as exc:
        errors.append(f"prompt component validation failed: {exc}")

    prompt_provenance = manifest.get("prompt_provenance")
    if not isinstance(prompt_provenance, dict) or not prompt_provenance.get("status"):
        errors.append("prompt_provenance status is missing")
    elif prompt_provenance.get("status") != "exact":
        message = (
            "Project Genie prompts are "
            f"{prompt_provenance.get('status')}, not exact paper provenance"
        )
        (errors if strict else warnings).append(message)

    compiled: dict[Path, Any] = {}
    for case in cases:
        trajectory_path = (root / str(case["trajectory"])).resolve()
        try:
            item = compiled.setdefault(trajectory_path, compile_trajectory(trajectory_path))
        except Exception as exc:
            errors.append(f"{case['case_id']}: {exc}")
            continue
        if item.alignment_status != "exact":
            message = f"{case['case_id']}: trajectory is {item.alignment_status}, not exact paper provenance"
            (errors if strict else warnings).append(message)
        first_frame = case.get("first_frame")
        if not first_frame:
            message = f"{case['case_id']}: first_frame is missing"
            (errors if strict else warnings).append(message)
        elif not (root / str(first_frame)).resolve().is_file():
            errors.append(f"{case['case_id']}: first_frame does not exist: {first_frame}")
        prompt_source = case.get("prompt_source")
        if not prompt_source or not (root / str(prompt_source)).resolve().is_file():
            errors.append(f"{case['case_id']}: prompt_source is missing or does not exist")
        if case.get("table") == 7:
            suffix = str(case["case_id"]).rsplit("_", 1)[-1]
            expected_orientation = {"in": "inward", "out": "outward"}.get(suffix)
            if case.get("paper_orientation") != expected_orientation:
                errors.append(f"{case['case_id']}: paper_orientation does not match In/Out column")
            if case.get("orientation_frame_status") != "exact":
                message = (
                    f"{case['case_id']}: orientation-specific conditioning frame is "
                    f"{case.get('orientation_frame_status', 'missing')}"
                )
                (errors if strict else warnings).append(message)
        if item.project_genie_status == "unsupported":
            message = f"{case['case_id']}: Project Genie cannot express this action space"
            (errors if strict else warnings).append(message)
        elif item.project_genie_status != "calibrated":
            message = f"{case['case_id']}: Project Genie adapter is {item.project_genie_status}"
            (errors if strict else warnings).append(message)

    report = {
        "ok": not errors,
        "strict": strict,
        "case_counts": counts,
        "trajectory_count": len(compiled),
        "paper_baseline": baseline_path.relative_to(root).as_posix(),
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "compile"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path, help="write the audit report as JSON")
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit(root, strict=args.strict)
    if args.command == "compile" and report["ok"]:
        output = (args.output_dir or (root / "compiled")).resolve()
        sources = sorted({(root / case["trajectory"]).resolve() for case in load_cases(root)["cases"]})
        report["compiled"] = [write_compiled_trajectory(source, output) for source in sources]
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
