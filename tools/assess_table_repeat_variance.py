#!/usr/bin/env python3
"""Apply the pre-registered five-to-thirty-run variance and stability gate."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy import stats


PERCENT_METRICS = {
    "exact_match_pct",
    "error_rate_pct",
    "depth_absrel_pct",
    "pairwise_distance_mae_pct",
    "object_permanence_pct",
    "character_consistency_pct",
}
UNBOUNDED_METRICS = {"dtw_m_mean", "fde_m_mean"}
METRICS = PERCENT_METRICS | UNBOUNDED_METRICS


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (count - rank) * pvalues[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def assess(values: list[float], metric: str) -> dict[str, Any]:
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    cv = sample_std / abs(mean) if abs(mean) > 1e-12 else math.inf
    if metric in PERCENT_METRICS:
        expand = sample_std > 5.0
        reason = "sample_std_gt_5_percentage_points" if expand else None
    else:
        expand = sample_std > 0.05 and cv > 0.20
        reason = "sample_std_gt_0.05_and_cv_gt_0.20" if expand else None
    result: dict[str, Any] = {
        "n": len(values),
        "mean": mean,
        "sample_std": sample_std,
        "coefficient_of_variation": cv,
        "expand_to_30": expand,
        "expansion_reason": reason,
    }
    if len(values) == 30:
        sem = sample_std / math.sqrt(30)
        critical = float(stats.t.ppf(0.975, 29))
        first, confirmation = values[:5], values[5:]
        welch = stats.ttest_ind(first, confirmation, equal_var=False)
        brown_forsythe = stats.levene(first, confirmation, center="median")
        welch_stat, welch_p = float(welch.statistic), float(welch.pvalue)
        if not math.isfinite(welch_p):
            same = all(value == first[0] for value in first + confirmation)
            welch_stat, welch_p = (0.0, 1.0) if same else (None, 0.0)
        bf_stat, bf_p = float(brown_forsythe.statistic), float(brown_forsythe.pvalue)
        if not math.isfinite(bf_p):
            # When both cohorts are constant there is no observed dispersion
            # difference for the variance test, regardless of their means.
            bf_stat, bf_p = 0.0, 1.0
        result.update(
            mean_ci95=[mean - critical * sem, mean + critical * sem],
            first5={"mean": statistics.fmean(first), "sample_std": statistics.stdev(first)},
            confirmation25={"mean": statistics.fmean(confirmation), "sample_std": statistics.stdev(confirmation)},
            welch_t={"statistic": welch_stat, "pvalue": welch_p},
            brown_forsythe={"statistic": bf_stat, "pvalue": bf_p},
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric",
        action="append",
        choices=sorted(METRICS),
        help="Restrict the gate to a table-applicable metric (repeatable).",
    )
    args = parser.parse_args()
    rows = []
    for line in args.results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("model") == args.model and row.get("case_id") == args.case_id and row.get("dimension") == args.dimension:
            rows.append(row)
    rows.sort(key=lambda row: int(str(row["seed"]).removeprefix("run")))
    if len(rows) not in {5, 30} or any(row.get("status") != "completed" for row in rows):
        raise ValueError(f"expected exactly 5 or 30 completed rows, got {len(rows)}")
    by_metric: dict[str, list[float]] = {}
    unavailable_metrics: dict[str, dict[str, Any]] = {}
    requested_metrics = args.metric or sorted(METRICS)
    for metric in requested_metrics:
        values = [_numeric(row.get("metrics", {}).get(metric)) for row in rows]
        if all(value is not None for value in values):
            by_metric[metric] = [float(value) for value in values]
        else:
            unavailable_metrics[metric] = {
                "status": "undefined",
                "reason": "non_numeric_or_missing_values",
                "observed_numeric_count": sum(value is not None for value in values),
                "required_repeat_count": len(rows),
                "missing_repeats": [
                    str(row["seed"])
                    for row, value in zip(rows, values, strict=True)
                    if value is None
                ],
            }
    if not by_metric:
        raise ValueError("no complete applicable table metrics found")
    metrics = {name: assess(values, name) for name, values in by_metric.items()}
    if len(rows) == 30:
        for test in ("welch_t", "brown_forsythe"):
            adjusted = _holm({name: result[test]["pvalue"] for name, result in metrics.items()})
            for name, pvalue in adjusted.items():
                metrics[name][test]["holm_pvalue"] = pvalue
    report = {
        "schema_version": 1,
        "model": args.model,
        "case_id": args.case_id,
        "dimension": args.dimension,
        "metrics_requested": requested_metrics,
        "unavailable_metrics": unavailable_metrics,
        "repeat_count": len(rows),
        "expand_group_to_30": any(result["expand_to_30"] for result in metrics.values()),
        "rule": {
            "percentage_metrics": "sample_std > 5 percentage points",
            "unbounded_metrics": "sample_std > 0.05 and coefficient_of_variation > 0.20",
            "confirmation_comparison": "runs 0-4 versus independent runs 5-29",
            "tests": "two-sided Welch t and median-centered Levene/Brown-Forsythe; Holm within group",
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
