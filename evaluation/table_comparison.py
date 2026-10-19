"""Published-versus-rerun aggregation for the paper's Tables 4 and 7."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

import yaml


class ComparisonError(ValueError):
    """Raised when results cannot support a paper-aligned comparison."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_baseline(
    path: Path,
    *,
    verify_sources: bool = True,
    missing_sources: str = "error",
    warn: Any = None,
) -> dict[str, Any]:
    """Load the paper baseline, checking its paper-source hashes.

    The sources are the paper's LaTeX tables, which live in a sibling
    checkout that public users do not have. ``missing_sources="warn"`` still
    hash-checks every source that is present but only warns (via ``warn``,
    default stderr) about absent ones; ``"error"`` (default) fails closed.
    """
    if missing_sources not in ("error", "warn"):
        raise ValueError(f"missing_sources must be 'error' or 'warn', got {missing_sources!r}")
    baseline = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or baseline.get("schema_version") != 1:
        raise ComparisonError(f"{path}: unsupported baseline schema")
    if baseline.get("aggregation", {}).get("generation_count") != 5:
        raise ComparisonError("paper baseline must specify five generations")
    if baseline.get("aggregation", {}).get("ddof") != 1:
        raise ComparisonError("paper baseline must specify sample standard deviation (ddof=1)")
    if verify_sources:
        for label, source in baseline.get("sources", {}).items():
            source_path = (path.parent / source["path"]).resolve()
            if not source_path.is_file():
                if missing_sources == "warn":
                    message = (
                        f"paper source {label} not present ({source['path']}); "
                        "using the baseline's recorded values unverified"
                    )
                    if warn is None:
                        import sys

                        print(f"warning: {message}", file=sys.stderr)
                    else:
                        warn(message)
                    continue
                raise ComparisonError(f"paper source {label} does not exist: {source_path}")
            actual = sha256(source_path)
            if actual != source["sha256"]:
                raise ComparisonError(
                    f"paper source {label} hash changed: expected {source['sha256']}, got {actual}"
                )
    return baseline


def result_template(baseline: Mapping[str, Any]) -> dict[str, Any]:
    repeats = [f"run{index}" for index in range(5)]
    results: dict[str, Any] = {}
    for table in (4, 7):
        for case_id, metrics in baseline["tables"][table]["cases"].items():
            results[case_id] = {
                metric: {repeat: None for repeat in repeats}
                for metric, published in metrics.items()
                if published is not None
            }
    return {
        "schema_version": 1,
        "model": baseline["model"],
        "repeat_ids": repeats,
        "results": results,
    }


def compare_results(
    baseline: Mapping[str, Any], rerun: Mapping[str, Any], *, allow_incomplete: bool = False
) -> dict[str, Any]:
    repeat_ids = [f"run{index}" for index in range(5)]
    if rerun.get("repeat_ids") != repeat_ids:
        raise ComparisonError(f"rerun repeat_ids must be exactly {repeat_ids}")
    rerun_results = rerun.get("results")
    if not isinstance(rerun_results, dict):
        raise ComparisonError("rerun results must be a mapping")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for table in (4, 7):
        for case_id, metrics in baseline["tables"][table]["cases"].items():
            supplied_metrics = rerun_results.get(case_id, {})
            for metric, published in metrics.items():
                metric_spec = baseline["metrics"][metric]
                row: dict[str, Any] = {
                    "table": table,
                    "case_id": case_id,
                    "metric": metric,
                    "metric_label": metric_spec["label"],
                    "direction": metric_spec["direction"],
                    "unit": metric_spec["unit"],
                    "published_mean": published[0] if published is not None else None,
                    "published_std": published[1] if published is not None else None,
                }
                if published is None:
                    row.update({"status": "not_applicable", "values": None})
                    rows.append(row)
                    continue
                run_map = supplied_metrics.get(metric, {}) if isinstance(supplied_metrics, dict) else {}
                values = [run_map.get(repeat) for repeat in repeat_ids] if isinstance(run_map, dict) else []
                valid = (
                    len(values) == 5
                    and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
                    and all(math.isfinite(float(value)) for value in values)
                )
                if not valid:
                    missing.append(f"{case_id}:{metric}")
                    row.update({"status": "pending", "values": values or None})
                    rows.append(row)
                    continue
                numeric = [float(value) for value in values]
                mean = statistics.fmean(numeric)
                std = statistics.stdev(numeric)
                delta = mean - float(published[0])
                relative = delta / abs(float(published[0])) if float(published[0]) else None
                row.update(
                    {
                        "status": "complete",
                        "values": dict(zip(repeat_ids, numeric)),
                        "rerun_mean": mean,
                        "rerun_std": std,
                        "delta": delta,
                        "relative_delta": relative,
                    }
                )
                rows.append(row)
    if missing and not allow_incomplete:
        raise ComparisonError(
            f"missing five numeric runs for {len(missing)} applicable cells: " + ", ".join(missing[:8])
        )
    return {
        "schema_version": 1,
        "model": rerun.get("model") or baseline["model"],
        "baseline_model": baseline["model"],
        "generation_count": 5,
        "ddof": 1,
        "complete": not missing,
        "pending_count": len(missing),
        "rows": rows,
        "paper_conflicts": baseline.get("paper_conflicts", []),
    }


def _number(value: Any) -> str:
    return "" if value is None else f"{float(value):.4g}"


def write_comparison(comparison: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = [
        "table", "case_id", "metric", "direction", "unit", "status",
        "published_mean", "published_std", "rerun_mean", "rerun_std", "delta", "relative_delta",
    ]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison["rows"])
    lines = [
        f"# {comparison['model']} vs. published {comparison.get('baseline_model', comparison['model'])}: "
        "Table 4/7 cross-comparison",
        "",
        f"Five-run sample statistics (`ddof=1`). Complete: `{str(comparison['complete']).lower()}`.",
    ]
    for table in (4, 7):
        lines.extend(
            [
                "",
                f"## Table {table}",
                "",
                "| Case | Metric | Published | Rerun | Delta | Status |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in comparison["rows"]:
            if row["table"] != table:
                continue
            published = "N/A" if row["published_mean"] is None else (
                f"{_number(row['published_mean'])} ± {_number(row['published_std'])}"
            )
            rerun_text = "—" if row.get("rerun_mean") is None else (
                f"{_number(row['rerun_mean'])} ± {_number(row['rerun_std'])}"
            )
            lines.append(
                f"| `{row['case_id']}` | {row['metric_label']} ({row['direction']}) | "
                f"{published} | {rerun_text} | {_number(row.get('delta')) or '—'} | {row['status']} |"
            )
    lines.extend(["", "## Preserved paper conflicts", ""])
    for conflict in comparison.get("paper_conflicts", []):
        lines.append(f"- `{conflict['id']}`: {conflict.get('status', 'unresolved')}")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
