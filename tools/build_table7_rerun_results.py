#!/usr/bin/env python3
"""Translate five WMEval Lite rows into the Table 4/7 comparison schema."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.table_comparison import load_baseline, result_template, sha256

DEFAULT_BASELINE = REPO_ROOT / "experiments" / "tables_4_7" / "paper_baseline.yaml"
METRIC_MAP = {
    "depth_absrel_pct": "depth_absrel_pct",
    "pairwise_distance_mae_pct": "dist_mae_pct",
    "object_permanence_pct": "object_permanence_pct",
    "character_consistency_pct": "character_consistency_pct",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def build_results(
    *,
    baseline_path: Path,
    results_jsonl: Path,
    evaluation_inputs_path: Path,
    source_case_id: str,
    target_case_id: str,
    model: str = "project_genie",
) -> dict[str, Any]:
    baseline_path = baseline_path.resolve()
    results_jsonl = results_jsonl.resolve()
    evaluation_inputs_path = evaluation_inputs_path.resolve()
    baseline = load_baseline(baseline_path)
    output = result_template(baseline)
    evaluation_inputs = json.loads(evaluation_inputs_path.read_text(encoding="utf-8"))
    if evaluation_inputs.get("status") != "ready":
        raise ValueError("evaluation input receipt is not ready")
    expected_videos = {
        run["repeat_id"]: (Path(run["candidate_video"]).resolve(), run["candidate_video_sha256"])
        for run in evaluation_inputs.get("runs", [])
    }
    repeat_ids = output["repeat_ids"]
    if set(expected_videos) != set(repeat_ids):
        raise ValueError("evaluation input receipt does not contain exactly run0 through run4")

    selected: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(results_jsonl):
        if (
            row.get("dimension") == "environment_object_consistency"
            and row.get("case_id") == source_case_id
            and row.get("model") == model
        ):
            seed = str(row.get("seed"))
            if seed in selected:
                raise ValueError(f"duplicate result row for {seed}")
            selected[seed] = row
    if set(selected) != set(repeat_ids):
        raise ValueError(
            f"result rows must contain exactly {repeat_ids}, got {sorted(selected)}"
        )

    target = output["results"].get(target_case_id)
    if not isinstance(target, dict):
        raise ValueError(f"target case is not in the paper baseline: {target_case_id}")
    incomplete_metrics: list[str] = []
    evaluator_locks: dict[str, dict[str, Any]] = {}
    devices: dict[str, str] = {}
    depth_model_selection: dict[str, str | None] = {}
    depth_model_used: dict[str, str | None] = {}
    for repeat_id in repeat_ids:
        row = selected[repeat_id]
        if row.get("status") != "completed":
            raise ValueError(f"{repeat_id}: evaluation status is {row.get('status')!r}")
        expected_path, expected_hash = expected_videos[repeat_id]
        candidate = Path(str(row.get("candidate_video"))).resolve()
        if candidate != expected_path or not candidate.is_file() or sha256(candidate) != expected_hash:
            raise ValueError(f"{repeat_id}: result candidate does not match evaluation receipt")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{repeat_id}: result has no metrics object")
        evaluator_lock = row.get("evaluator_lock")
        if not isinstance(evaluator_lock, dict):
            raise ValueError(f"{repeat_id}: result has no evaluator lock")
        evaluator_locks[repeat_id] = evaluator_lock
        full_report = row.get("full_report")
        if not isinstance(full_report, dict) or not isinstance(
            full_report.get("device"), str
        ):
            raise ValueError(f"{repeat_id}: result has no evaluation device")
        devices[repeat_id] = str(full_report["device"])
        module_report = (
            full_report.get("results", {}).get("environment_object_consistency", {})
        )
        if not isinstance(module_report, dict):
            module_report = {}
        selection = module_report.get("depth_model_selection")
        used = module_report.get("depth_model_used")
        depth_model_selection[repeat_id] = (
            str(selection) if selection is not None else None
        )
        depth_model_used[repeat_id] = str(used) if used is not None else None
        for source_metric, target_metric in METRIC_MAP.items():
            if target_metric not in target:
                continue
            value = metrics.get(source_metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                target[target_metric][repeat_id] = float(value)
            else:
                incomplete_metrics.append(f"{repeat_id}:{target_metric}")

    lock_values = {json.dumps(value, sort_keys=True) for value in evaluator_locks.values()}
    if len(lock_values) != 1:
        raise ValueError("five-run results do not share one evaluator lock")
    if len(set(devices.values())) != 1:
        raise ValueError("five-run results do not share one evaluation device")

    output["provenance"] = {
        "claim_scope": evaluation_inputs.get("claim_scope"),
        "source_case_id": source_case_id,
        "target_case_id": target_case_id,
        "results_jsonl": str(results_jsonl),
        "results_jsonl_sha256": sha256(results_jsonl),
        "evaluation_inputs": str(evaluation_inputs_path),
        "evaluation_inputs_sha256": sha256(evaluation_inputs_path),
        "incomplete_metrics": incomplete_metrics,
        "evaluation_settings": {
            "evaluator_lock": evaluator_locks[repeat_ids[0]],
            "device": devices[repeat_ids[0]],
            "depth_model_selection_by_repeat": depth_model_selection,
            "depth_model_used_by_repeat": depth_model_used,
        },
        "parity_limitations": evaluation_inputs.get("parity_limitations", []),
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--evaluation-inputs", type=Path, required=True)
    parser.add_argument("--source-case-id", default="WanderingGarden/scene_first")
    parser.add_argument("--target-case-id", default="table7/WanderingGarden/360_in")
    parser.add_argument("--model", default="project_genie")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_results(
        baseline_path=args.baseline,
        results_jsonl=args.results_jsonl,
        evaluation_inputs_path=args.evaluation_inputs,
        source_case_id=args.source_case_id,
        target_case_id=args.target_case_id,
        model=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
