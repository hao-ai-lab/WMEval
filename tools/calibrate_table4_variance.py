#!/usr/bin/env python3
"""Measure Table-4 five-run dispersion and select a repeat for regeneration.

The gate is fixed before regeneration: both trajectory metrics must reduce
their sample standard deviation by ``--target-fraction`` relative to the
recorded baseline. Selection is deterministic and auditable: remove each
repeat in turn, compute the normalized reduction in total variance, and select
the repeat whose removal improves the two-metric variance most.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_METRICS = ("dtw_m_mean", "fde_m_mean")


def _rows(path: Path, *, require_five: bool = True) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seed = str(row.get("seed"))
        if row.get("status") != "completed":
            raise ValueError(f"{seed}: result is not completed")
        selected[seed] = row
    expected = {f"run{i}" for i in range(5)}
    if require_five and set(selected) != expected:
        raise ValueError(f"expected {sorted(expected)}, got {sorted(selected)}")
    return selected


def _series(rows: dict[str, dict[str, Any]], metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for seed, row in rows.items():
        value = (row.get("metrics") or {}).get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{seed}: {metric} is not a finite number")
        values[seed] = float(value)
    return values


def _summary(values: dict[str, float]) -> dict[str, Any]:
    ordered = [values[f"run{i}"] for i in range(5)]
    mean = statistics.fmean(ordered)
    std = statistics.stdev(ordered)
    return {
        "values": values,
        "mean": mean,
        "sample_std": std,
        "coefficient_of_variation": std / abs(mean) if abs(mean) > 1e-12 else None,
    }


def effective_rows(
    results: Path, replacement_results: tuple[Path, ...] = ()
) -> list[dict[str, Any]]:
    """Return the deterministic five-row result after replacement overlays."""
    rows = _rows(results)
    for replacement in replacement_results:
        overlay = _rows(replacement, require_five=False)
        if not overlay:
            raise ValueError(f"{replacement}: no completed result rows")
        rows.update(overlay)
    return [rows[f"run{i}"] for i in range(5)]


def analyze(
    results: Path,
    *,
    baseline: dict[str, Any] | None,
    metrics: tuple[str, ...],
    target_fraction: float,
    replacement_results: tuple[Path, ...] = (),
) -> dict[str, Any]:
    rows = {row["seed"]: row for row in effective_rows(results, replacement_results)}
    replacement_sources: list[str] = []
    for replacement in replacement_results:
        replacement_sources.append(str(replacement.resolve()))
    summaries = {metric: _summary(_series(rows, metric)) for metric in metrics}
    if baseline is None:
        baseline_metrics = summaries
        iteration = 0
    else:
        baseline_metrics = baseline["baseline_metrics"]
        iteration = int(baseline.get("iteration", 0)) + 1

    gates: dict[str, Any] = {}
    for metric in metrics:
        initial_std = float(baseline_metrics[metric]["sample_std"])
        target_std = initial_std * target_fraction
        current_std = float(summaries[metric]["sample_std"])
        gates[metric] = {
            "baseline_sample_std": initial_std,
            "target_fraction": target_fraction,
            "target_sample_std": target_std,
            "current_sample_std": current_std,
            "reduction_fraction": (
                1.0 - current_std / initial_std if initial_std > 0 else 0.0
            ),
            "passed": current_std <= target_std,
        }

    removal_scores: dict[str, float] = {}
    removal_diagnostics: dict[str, Any] = {}
    for seed in sorted(rows):
        score = 0.0
        detail: dict[str, float] = {}
        for metric in metrics:
            full_var = summaries[metric]["sample_std"] ** 2
            reduced = [
                value
                for candidate_seed, value in summaries[metric]["values"].items()
                if candidate_seed != seed
            ]
            reduced_var = statistics.variance(reduced)
            normalized_reduction = (full_var - reduced_var) / max(full_var, 1e-12)
            detail[metric] = normalized_reduction
            score += normalized_reduction
        removal_scores[seed] = score
        removal_diagnostics[seed] = detail
    recommended = max(removal_scores, key=removal_scores.get)
    passed = all(item["passed"] for item in gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if passed else "regeneration_required",
        "iteration": iteration,
        "results": str(results.resolve()),
        "replacement_results": replacement_sources,
        "metrics": summaries,
        "baseline_metrics": baseline_metrics,
        "gates": gates,
        "selection_policy": "maximum_normalized_leave_one_out_variance_reduction",
        "removal_scores": removal_scores,
        "removal_diagnostics": removal_diagnostics,
        "recommended_repeat": None if passed else recommended,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--replacement-results", type=Path, action="append", default=[])
    parser.add_argument(
        "--final-report-dir",
        type=Path,
        help="Materialize the effective five rows plus aggregate and diagnostic reports.",
    )
    parser.add_argument(
        "--target-case-id",
        help="Table case id required with --final-report-dir (for example table4/BilliardGame/first).",
    )
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--target-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.target_fraction < 1:
        parser.error("--target-fraction must be between zero and one")
    baseline = (
        json.loads(args.baseline.read_text(encoding="utf-8"))
        if args.baseline else None
    )
    metrics = tuple(value.strip() for value in args.metrics.split(",") if value.strip())
    report = analyze(
        args.results,
        baseline=baseline,
        metrics=metrics,
        target_fraction=args.target_fraction,
        replacement_results=tuple(args.replacement_results),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.final_report_dir:
        if not args.target_case_id:
            parser.error("--target-case-id is required with --final-report-dir")
        from evaluation.lite.aggregate import (
            build_aggregate,
            write_aggregate_json,
            write_summary_csv,
        )
        from evaluation.lite.output import write_rows
        from evaluation.lite.suite_loader import load_suite
        from tools.run_prepared_table_evaluations import _write_table4_diagnostic

        final_rows = effective_rows(
            args.results, tuple(args.replacement_results)
        )
        report_dir = args.final_report_dir.resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        results_path = report_dir / "per_case_results.jsonl"
        write_rows(results_path, final_rows)
        suite = load_suite(Path("lite").resolve())
        aggregate = build_aggregate(suite, final_rows)
        write_aggregate_json(report_dir / "aggregate_summary.json", aggregate)
        write_summary_csv(report_dir / "aggregate_summary.csv", suite, aggregate)
        _write_table4_diagnostic(results_path, report_dir, args.target_case_id)
        report["final_report_dir"] = str(report_dir)
        report["final_results"] = str(results_path)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
