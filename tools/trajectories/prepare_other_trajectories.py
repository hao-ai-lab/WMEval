#!/usr/bin/env python3
"""Compile supplemental trajectories for Lite cases with documented schedules."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.trajectories import write_compiled_trajectory

DEFAULT_ROOT = REPO_ROOT / "experiments" / "trajectory_reconstructions"


def compile_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{manifest_path}: unsupported manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{manifest_path}: cases must be a non-empty list")
    output = root / "compiled"
    rows = []
    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError(f"{manifest_path}: missing or duplicate case_id {case_id!r}")
        seen.add(case_id)
        source = (root / str(case.get("trajectory", ""))).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        compiled = write_compiled_trajectory(source, output)
        rows.append(
            {
                "case_id": case_id,
                "provenance": case.get("provenance"),
                "trajectory": source.relative_to(root).as_posix(),
                "compiled": compiled,
            }
        )
    report = {
        "schema_version": 1,
        "status": "ready",
        "case_count": len(rows),
        "cases": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    report = compile_manifest(args.root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
