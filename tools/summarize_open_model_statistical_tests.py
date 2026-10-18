#!/usr/bin/env python3
"""Summarize all preregistered five/30-run gates and statistical comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.assess_table7_paired_character_variance import required_pair_count
from tools.audit_open_model_table_completion import ACTIVE_CASES
from tools.export_open_model_tables_latex import required_repeat_count


def directional_path(model: str, case_id: str, repeat_count: int) -> Path:
    table, scene, variant = case_id.split("/")
    cohort = "30_run" if repeat_count == 30 else "five_run"
    filename = "variance_comparison.json" if repeat_count == 30 else "variance_gate.json"
    return ROOT / "reports" / table / model / scene / variant / cohort / filename


def paired_path(model: str, scene: str, regime: str, repeat_count: int) -> Path:
    cohort = "30_run" if repeat_count == 30 else "five_run"
    filename = "variance_comparison.json" if repeat_count == 30 else "variance_gate.json"
    return ROOT / "reports/table7" / model / scene / "paired_character" / regime / cohort / filename


def numeric_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def flatten_report(
    *, model: str, group: str, kind: str, path: Path, repeat_count: int
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != 1
        or report.get("model") != model
        or report.get("repeat_count") != repeat_count
        or not isinstance(report.get("metrics"), dict)
        or not report["metrics"]
    ):
        raise ValueError(f"invalid statistical report: {path}")
    rows = []
    for metric, value in sorted(report["metrics"].items()):
        row: dict[str, Any] = {
            "model": model,
            "group": group,
            "kind": kind,
            "metric": metric,
            "repeat_count": repeat_count,
            "mean": numeric_or_none(value.get("mean")),
            "sample_std": numeric_or_none(value.get("sample_std")),
            "ci95_low": None,
            "ci95_high": None,
            "first5_mean": None,
            "confirmation25_mean": None,
            "welch_pvalue": None,
            "welch_holm_pvalue": None,
            "brown_forsythe_pvalue": None,
            "brown_forsythe_holm_pvalue": None,
            "holm_significant": False,
        }
        if row["mean"] is None or row["sample_std"] is None:
            raise ValueError(f"non-finite mean/std in {path}: {metric}")
        if repeat_count == 30:
            ci = value.get("mean_ci95")
            welch = value.get("welch_t", {})
            brown = value.get("brown_forsythe", {})
            if not isinstance(ci, list) or len(ci) != 2:
                raise ValueError(f"missing 95% CI in {path}: {metric}")
            row.update(
                ci95_low=numeric_or_none(ci[0]),
                ci95_high=numeric_or_none(ci[1]),
                first5_mean=numeric_or_none(value.get("first5", {}).get("mean")),
                confirmation25_mean=numeric_or_none(value.get("confirmation25", {}).get("mean")),
                welch_pvalue=numeric_or_none(welch.get("pvalue")),
                welch_holm_pvalue=numeric_or_none(welch.get("holm_pvalue")),
                brown_forsythe_pvalue=numeric_or_none(brown.get("pvalue")),
                brown_forsythe_holm_pvalue=numeric_or_none(brown.get("holm_pvalue")),
            )
            required = (
                "ci95_low",
                "ci95_high",
                "first5_mean",
                "confirmation25_mean",
                "welch_pvalue",
                "welch_holm_pvalue",
                "brown_forsythe_pvalue",
                "brown_forsythe_holm_pvalue",
            )
            if any(row[name] is None for name in required):
                raise ValueError(f"incomplete 30-run comparison in {path}: {metric}")
            row["holm_significant"] = bool(
                row["welch_holm_pvalue"] < 0.05
                or row["brown_forsythe_holm_pvalue"] < 0.05
            )
        rows.append(row)
    return rows


def collect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, cases in ACTIVE_CASES.items():
        for case_id in cases:
            count = required_repeat_count(model, case_id)
            rows.extend(
                flatten_report(
                    model=model,
                    group=case_id,
                    kind="directional_cell",
                    path=directional_path(model, case_id, count),
                    repeat_count=count,
                )
            )
    for model in ACTIVE_CASES:
        for scene in ("BilliardGame", "MirrorRollingBall"):
            for regime in ("perturb", "360"):
                count = required_pair_count(model, scene, regime)
                rows.extend(
                    flatten_report(
                        model=model,
                        group=f"table7/{scene}/{regime}_paired_character",
                        kind="reported_paired_character_cell",
                        path=paired_path(model, scene, regime, count),
                        repeat_count=count,
                    )
                )
    return rows


def write_summary(output_dir: Path) -> dict[str, Any]:
    rows = collect()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "statistical_comparisons.csv"
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "policy": {
            "screen": "sample SD > 5 percentage points; DTW/FDE additionally require CV > 0.20 and SD > 0.05",
            "confirmation": "original runs 0-4 versus independent runs 5-29",
            "tests": "two-sided Welch t and median-centered Levene/Brown-Forsythe with Holm correction within group",
        },
        "row_count": len(rows),
        "thirty_run_metric_count": sum(row["repeat_count"] == 30 for row in rows),
        "holm_significant_count": sum(row["holm_significant"] for row in rows),
        "holm_significant_rows": [row for row in rows if row["holm_significant"]],
        "rows": rows,
        "csv": str(csv_path),
    }
    json_path = output_dir / "statistical_comparisons.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/tables_4_7/final")
    args = parser.parse_args()
    try:
        report = write_summary(args.output_dir)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "incomplete", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
