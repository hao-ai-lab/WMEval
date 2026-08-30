"""Unified per-case JSONL output: row schema, incremental writer, resume keys.

Every evaluation of one ``(dimension, case, model, seed)`` work item produces
exactly one JSON line. Rows are appended as soon as they are produced so a
crashed or interrupted run keeps its finished work; ``--resume`` skips work
items whose key already appears in the output file (same mechanism as the
vlm_consistency batch pipeline).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROW_SCHEMA_VERSION = 1

PER_CASE_RESULTS_FILENAME = "per_case_results.jsonl"
AGGREGATE_JSON_FILENAME = "aggregate_summary.json"
AGGREGATE_CSV_FILENAME = "aggregate_summary.csv"

RowKey = tuple[str, str, str, str]


def row_key(row: Mapping[str, Any]) -> RowKey:
    return (
        str(row.get("dimension") or ""),
        str(row.get("case_id") or ""),
        str(row.get("model") or ""),
        str(row.get("seed") or ""),
    )


def build_row(
    *,
    suite_name: str,
    suite_version: str,
    run_id: str,
    dimension: str,
    case_id: str,
    scene: str,
    model: str,
    seed: str,
    status: str,
    candidate_video: str | None,
    reference_video: str | None,
    evaluator_lock: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    full_report: Mapping[str, Any] | None = None,
    error: str | None = None,
    timing_sec: float | None = None,
) -> dict[str, Any]:
    lock_summary = None
    if evaluator_lock is not None:
        lock_summary = {
            "release_id": evaluator_lock.get("release_id"),
            "evaluator_fingerprint": evaluator_lock.get("evaluator_fingerprint"),
        }
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "suite_name": suite_name,
        "suite_version": suite_version,
        "run_id": run_id,
        "dimension": dimension,
        "case_id": case_id,
        "scene": scene,
        "model": model,
        "seed": seed,
        "status": status,
        "error": error,
        "candidate_video": candidate_video,
        "reference_video": reference_video,
        "evaluator_lock": lock_summary,
        "metrics": dict(metrics) if metrics is not None else {},
        "full_report": dict(full_report) if full_report is not None else None,
        "timing_sec": timing_sec,
    }


def append_row(path: str | Path, row: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}: line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}: line {line_number}: row must be an object")
            rows.append(row)
    return rows


def load_done_keys(path: str | Path) -> set[RowKey]:
    return {row_key(row) for row in read_rows(path)}
