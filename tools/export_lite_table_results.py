#!/usr/bin/env python3
"""Export WMEval Lite results for the Table 4/7 cells into the paper-comparison schema.

The Table 4/7 cells are ordinary Lite cases (``tools/build_table_lite_cases.py``),
so a model is evaluated with the usual runner::

    python tools/run_wmeval_lite.py run --suite lite \\
      --model-root runs/<model> \\
      --dimensions environment_object_consistency,kinematics_mechanics \\
      --output-dir reports/wmeval_lite/<model>

This tool maps those rows back to their table cells (via each bundle's
``reference/artifacts/table_cell.json``) and writes a results file accepted by
``tools/compare_table_results.py compare``::

    python tools/export_lite_table_results.py \\
      --results reports/wmeval_lite/<model>/per_case_results.jsonl \\
      --model "<Display name>" --output reports/wmeval_lite/<model>/table_results.json
    python tools/compare_table_results.py compare \\
      --results reports/wmeval_lite/<model>/table_results.json --allow-incomplete
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.table_comparison import load_baseline, result_template  # noqa: E402

CASES_ROOT = ROOT / "lite/cases"
DEFAULT_BASELINE = ROOT / "experiments/tables_4_7/paper_baseline.yaml"
METRIC_MAP = {
    # Table 7 — environment_object_consistency
    "depth_absrel_pct": "depth_absrel_pct",
    "dist_mae_pct": "pairwise_distance_mae_pct",
    "object_permanence_pct": "object_permanence_pct",
    "character_consistency_pct": "character_consistency_pct",
    # Table 4 — kinematics_mechanics
    "dtw": "dtw_m_mean",
    "fde": "fde_m_mean",
    "error_rate_pct": "error_rate_pct",
    "exact_match_pct": "exact_match_pct",
}


def table_cell(case_id: str, cases_root: Path = CASES_ROOT) -> str | None:
    path = cases_root / case_id / "reference" / "artifacts" / "table_cell.json"
    if not path.is_file():
        return None
    return str(json.loads(path.read_text(encoding="utf-8"))["table_cell"])


def export(
    rows: list[dict[str, Any]],
    *,
    baseline_path: Path = DEFAULT_BASELINE,
    model: str | None = None,
    seed_order: list[str] | None = None,
    cases_root: Path = CASES_ROOT,
    missing_sources: str = "warn",
) -> dict[str, Any]:
    # Only the baseline's table shape is needed here; the paper LaTeX sources
    # it pins live in a separate checkout most users do not have.
    output = result_template(load_baseline(baseline_path, missing_sources=missing_sources))
    if not model:
        names = sorted({str(row["model"]) for row in rows if row.get("model")})
        model = ", ".join(names) or None
    if model:
        output["model"] = model
    repeats = output["repeat_ids"]
    by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        cell = table_cell(str(row.get("case_id")), cases_root)
        if cell is None or cell not in output["results"]:
            continue
        by_cell.setdefault(cell, {})[str(row["seed"])] = row.get("metrics") or {}
    for cell, seeds in by_cell.items():
        if seed_order:
            ordered = [seed for seed in seed_order if seed in seeds]
        elif all(seed in repeats for seed in seeds):
            ordered = [seed for seed in repeats if seed in seeds]
        else:
            ordered = sorted(seeds)
        for repeat, seed in zip(repeats, ordered):
            for target, source in METRIC_MAP.items():
                if target in output["results"][cell] and seeds[seed].get(source) is not None:
                    output["results"][cell][target][repeat] = float(seeds[seed][source])
    output["provenance"] = {
        "source": "wmeval_lite_table_cells",
        "claim_scope": "paper_protocol_reconstruction_not_exact_paper_parity",
        "cells": sorted(by_cell),
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="Lite per_case_results.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="display name written into the results file")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--seed-order", help="comma-separated seeds mapped to run0..run4 (default: run ids or sorted)")
    parser.add_argument(
        "--suite", type=Path, default=CASES_ROOT.parent,
        help="suite root whose cases/<id>/reference/artifacts/table_cell.json map rows "
        "to table cells, e.g. a downloaded dataset directory (default: %(default)s)",
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = export(rows, baseline_path=args.baseline, model=args.model,
                    seed_order=args.seed_order.split(",") if args.seed_order else None,
                    cases_root=args.suite / "cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": result["provenance"]["cells"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
